"""Test package.

Made a package (rather than loose modules) so every test can share
`tests.helpers`, which pins the device to CPU before keras is imported.
"""
