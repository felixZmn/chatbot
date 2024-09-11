import discord
import functools
import logging

from discord import Message
from discord.ext import commands
from src.ChatBot import ChatBot, Course
from src.discord.Dropdowns import DropdownView
from src.discord.disclaimer import disclaimer

chatbot_logger = logging.getLogger('ChatBot')


class DiscordBot(commands.Bot):
    chatbot: ChatBot = None

    def __init__(self, documents_dir: str, index_dir: str):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=commands.when_mentioned_or('$'), intents=intents)
        self.chatbot = ChatBot(
            documents_dir=documents_dir, index_dir=index_dir)

    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    async def on_message(self, message: Message):
        # ignore messages from the bot itself
        if message.author.id == self.user.id:
            return

        # only reply to DMs
        if not isinstance(message.channel, discord.channel.DMChannel):
            return

        # get pinned messages
        pinned_messages = await message.channel.pins()

        course: Course = None
        disclaimer_present = False
        # check if course is pinned
        for msg in pinned_messages:
            if msg.content.startswith('Kurs:') and msg.author.id == self.user.id:
                course = Course(msg.content.split(': ')[1].lower())
                break
            if msg.content == disclaimer:
                disclaimer_present = True

        if not disclaimer_present:
            msg = await message.channel.send(disclaimer)
            await msg.pin()

        if course is None:
            view = DropdownView()
            await message.channel.send('Bitte wähle deinen Kurs und stelle die Frage anschließend erneut', view=view)
        else:
            fun = functools.partial(
                self.chatbot.perform_query, message.content, course)
            response = await self.loop.run_in_executor(None, fun)
            await message.channel.send(response)
