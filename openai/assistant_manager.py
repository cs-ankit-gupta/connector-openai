"""
Copyright start
MIT License
Copyright (c) 2024 Fortinet Inc
Copyright end
"""

from .assistant_event_handler import EventHandler
from .operations import *

logger = get_logger(LOGGER_NAME)


class AssistantManager:

    def __init__(self, config, params):
        self.config = config
        self.params = params

    def get_llm_response(self):
        payload = {'thread_id': self.params['thread_id'], 'role': self.params['role'], 'content': self.params['content']}
        create_thread_message(config=self.config, params=payload)
        assistant_response = self.run_assistant()
        return assistant_response

    def run_assistant(self, instructions=""):
        client = openai.OpenAI(api_key=self.config['apiKey'], project=self.config.get('project'),
                               organization=self.config.get('organization'))
        event_handler = EventHandler(config=self.config, params=self.params)
        with client.beta.threads.runs.create_and_stream(
                thread_id=self.params['thread_id'],
                assistant_id=self.params['assistant_id'],
                instructions=instructions,
                event_handler=event_handler
        ) as stream:
            stream.until_done()
        return {"llm_response": event_handler.get_resposne(), "token_usage": event_handler.token_usage}


def get_llm_response(config, params):
    assistant = AssistantManager(config, params)
    return assistant.get_llm_response()
