# Chatbot

A Discord-based chatbot designed to answer university-related questions using LlamaIndex and Llama 3.1.

## Table of Contents

1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [Usage](#usage)

## Features

-   Answers university-related questions based on faculty guidelines and other documents
-   Integrates seamlessly with Discord for easy access
-   Utilizes LlamaIndex for efficient document indexing and retrieval
-   Powered by Llama 3.1 for natural language processing

## Technologies Used

-   Python
-   LlamaIndex
-   Llama 3.1
-   Discord.py

## Installation

The "prooven" way to install the chatbot is via ansible on a linux system. The
alternative way is to install the chatbot manually. Therefore, the playbook can be
used as a reference. The installation playbook is located in the `infrastructure`
directory.

### Step By Step Guide

1.  Installing ansible:

        ```bash
        sudo apt install ansible
        ```

    If the connection to the server is established via ssh useranme and password instead
    of ssh key, the package `sshpass` must be installed additionally:

```bash
sudo apt install sshpass
```

2. Clone the repository:

```bash
git clone git@github.com:felixZmn/chatbot.git
cd chatbot
```

3. Create an inventory file:

```yaml
ungrouped:
    vars:
        ansible_user: <username>
        # optional, if no ssh key is used
        ansible_ssh_pass: <password>
    hosts: 123.123.123.123
```

4. Execute the playbook:

```bash
ansible-playbook -i inventory.yml playbook.yml
```

5. Post-Installation steps

-   Create the discord bot in the Discord Developer Portal
-   Add the bot token to the `.env` file
-   Invite the bot to your Discord server using the OAuth2 URL generated in the Discord Developer Portal

## Usage

To start the chatbot, run the following command:

```bash
python DiscordChatBot.py
```

## Known Issues

-   Error loading "fbgemm.dll" or one of its dependencies:
    -   Solution: Install the Visual Studio Installer and the MSVC v143 VS Build Tools\
        Read more [here](https://github.com/pytorch/pytorch/issues/131662#issuecomment-2252589253)
