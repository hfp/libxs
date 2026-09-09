# Sync Barrier

Rendezvous and broadcast, checked rather than timed: a barrier that releases a
task early is fast and wrong, so what the test asserts is that no task reads a
value from the round before, and that two consecutive broadcasts do not share
storage.

```c
--8<-- "tests/sync_barrier.c"
```
