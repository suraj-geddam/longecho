import pytest
import asyncio
from longecho.voice_event_broadcaster import VoiceEventBroadcaster


@pytest.mark.asyncio
async def test_subscribe_creates_queue():
    """Subscribing should return an asyncio.Queue"""
    broadcaster = VoiceEventBroadcaster()
    queue = broadcaster.subscribe()

    assert isinstance(queue, asyncio.Queue)
    assert len(broadcaster.subscribers) == 1


@pytest.mark.asyncio
async def test_unsubscribe_removes_queue():
    """Unsubscribing should remove the queue from subscribers"""
    broadcaster = VoiceEventBroadcaster()
    queue = broadcaster.subscribe()

    broadcaster.unsubscribe(queue)

    assert len(broadcaster.subscribers) == 0


@pytest.mark.asyncio
async def test_broadcast_sends_to_all_subscribers():
    """Broadcasting should send event to all subscribed queues"""
    broadcaster = VoiceEventBroadcaster()
    queue1 = broadcaster.subscribe()
    queue2 = broadcaster.subscribe()

    event = {"type": "ready", "voice": "test"}
    broadcaster.broadcast(event)

    assert await queue1.get() == event
    assert await queue2.get() == event


@pytest.mark.asyncio
async def test_broadcast_with_no_subscribers():
    """Broadcasting with no subscribers should not raise"""
    broadcaster = VoiceEventBroadcaster()

    # Should not raise
    broadcaster.broadcast({"type": "ready", "voice": "test"})
