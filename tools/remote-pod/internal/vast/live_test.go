package vast

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestLive reads the account (credit, offers, instances) with the real API; it changes nothing. Runs only with
// VAST_LIVE=1 and a key in ~/.config/vastai/vast_api_key.
func TestLive(t *testing.T) {
	if os.Getenv("VAST_LIVE") != "1" {
		t.Skip("set VAST_LIVE=1")
	}
	home, _ := os.UserHomeDir()
	b, err := os.ReadFile(filepath.Join(home, ".config", "vastai", "vast_api_key"))
	if err != nil {
		t.Skip("no vast key")
	}
	c := New(strings.TrimSpace(string(b)))
	ctx := context.Background()
	credit, perHr, err := c.Balance(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("credit $%.2f, spending $%.3f/hr", credit, perHr)
	offers, err := c.Offers(ctx, OfferQuery{GPU: "RTX 4090", DiskGB: 80, MinReliability: 0.98, MinInetDown: 500, Limit: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(offers) == 0 {
		t.Fatal("no RTX 4090 offers")
	}
	for _, o := range offers {
		t.Logf("offer %d %s $%.3f/hr %.0f cores %.0f GB disk %.0f Mbps rel %.3f %s", o.ID, o.GPUName, o.Price,
			o.CPUCores, o.DiskSpace, o.InetDown, o.Reliability, o.Location)
	}
	list, err := c.Instances(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for _, in := range list {
		got, err := c.Get(ctx, in.ID)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("instance %d %s %s $%.3f/hr up %s ip %s port22->%d", got.ID, got.State(), got.GPUName, got.PerHour,
			got.Uptime(), got.PublicIP, got.HostPort(22))
	}
}
