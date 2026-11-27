package client

import (
	"context"
	"time"

	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/grant"
	"github.com/SamuelHaidu/rnn-research/tools/remote-pod/internal/proto"
)

// SignFunc decides and signs one of the pod's requests (a grant.Signer's Sign: the policy is checked on this side).
type SignFunc func(context.Context, grant.Request) (grant.Grant, error)

// ServeSigner answers the pod's grant requests over WS /v1/signer until ctx ends, reconnecting on its own (the agent
// keeps the newest connection). The store key never leaves this process: only URLs for single objects go back.
func (c *Client) ServeSigner(ctx context.Context, sign SignFunc, onState func(connected bool, err error)) {
	for ctx.Err() == nil {
		conn, err := c.dial("/v1/signer", nil)
		if err != nil {
			if onState != nil {
				onState(false, err)
			}
			select {
			case <-ctx.Done():
				return
			case <-time.After(5 * time.Second):
			}
			continue
		}
		if onState != nil {
			onState(true, nil)
		}
		stop := context.AfterFunc(ctx, func() { _ = conn.Close() })
		for {
			var req proto.SignRequest
			if err := conn.ReadJSON(&req); err != nil {
				break
			}
			rep := proto.SignReply{ID: req.ID}
			if g, err := sign(ctx, grant.Request{Op: req.Op, Key: req.Key, Size: req.Size}); err != nil {
				rep.Error = err.Error()
			} else {
				rep.URL, rep.Expires = g.URL, g.Expires
			}
			if err := conn.WriteJSON(rep); err != nil {
				break
			}
		}
		stop()
		_ = conn.Close()
		if onState != nil && ctx.Err() == nil {
			onState(false, nil)
		}
	}
}
