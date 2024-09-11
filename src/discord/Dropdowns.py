import discord

import src.ChatBot


class Dropdown(discord.ui.Select):
    def __init__(self):

        # Set the options that will be presented inside the dropdown
        options = []

        for course in src.ChatBot.Course:
            options.append(discord.SelectOption(
                label=course.value, description=course.value))

        # Dropdown
        super().__init__(placeholder='Bitte Kurs wählen...',
                         min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        # Sends a message with the course that was selected
        await interaction.response.send_message(f'Kurs: {self.values[0]}')
        message = await interaction.original_response()
        # pin the message for further use
        await message.pin()
        print(f"message: {message}")


class DropdownView(discord.ui.View):
    def __init__(self):
        super().__init__()

        # Adds the dropdown to our view object.
        self.add_item(Dropdown())
