from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from datetime import datetime
import asyncio

async def is_user_authorized(client, phone):
    if not await client.is_user_authorized():
        await client.send_code_request(phone)
        await client.sign_in(phone, input('Enter the code: '))

async def get_channel_entity(client, channel_username):
    return await client.get_entity(channel_username)

async def download_media(client, channel_entity, date_from):
    posts = await client(GetHistoryRequest(
        peer=channel_entity,
        limit=0,
        offset_date=None,
        offset_id=0,
        max_id=0,
        min_id=0,
        add_offset=0,
        hash=0
    ))

    for message in posts.messages:
        if message.media is not None and message.date > date_from:
            print(f"Downloading media from {message.date}")
            await client.download_media(message=message, file='path_where_you_want_to_store')
            await asyncio.sleep(1)

async def main():
    api_id = 'your_api_id'
    api_hash = 'your_api_hash'
    phone = 'your_phone_number'
    channel_username = 'channel_username'
    date_from = datetime(2023, 1, 1)

    async with TelegramClient('anon', api_id, api_hash) as client:
        await is_user_authorized(client, phone)
        channel_entity = await get_channel_entity(client, channel_username)
        await download_media(client, channel_entity, date_from)

if __name__ == '__main__':
    asyncio.run(main())
