import asyncio
from typing import Any


class VoiceEventBroadcaster:
    """
    Pub-sub broadcaster for voice events.

    Allows multiple SSE clients to subscribe and receive
    voice processing events (processing, ready, error).
    """

    def __init__(self):
        self.subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        """Create a new subscription queue and add it to subscribers."""
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a subscription queue from subscribers."""
        self.subscribers.discard(queue)

    def broadcast(self, event: dict[str, Any]) -> None:
        """Send event to all subscribed queues."""
        for queue in self.subscribers:
            queue.put_nowait(event)
