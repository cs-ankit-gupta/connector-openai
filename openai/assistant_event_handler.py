"""
Copyright start
MIT License
Copyright (c) 2024 Fortinet Inc
Copyright end
"""
from typing_extensions import override
from openai import AssistantEventHandler
from openai.types.beta.threads.runs import RunStepDelta
from openai.types.beta.threads import Message, MessageDelta
from openai.types.beta.threads.runs import ToolCall, RunStep
from openai.types.beta import AssistantStreamEvent
from .operations import *
from .utils import execute_connector_action

logger = get_logger(LOGGER_NAME)


class EventHandler(AssistantEventHandler):
    def __init__(self, config, params, last_message_id):
        super().__init__()
        self.run_id = None
        self.llm_response = []
        self.config = config
        self.params = params
        self.tool_outputs = []
        self.token_usage = {}
        self.function_call_token_usage = None
        self.last_message_id = last_message_id

    # Executes on every event
    @override
    def on_event(self, event: AssistantStreamEvent) -> None:
        logger.info(f'event: {event.event}')

    # thread.run.step.created
    @override
    def on_run_step_created(self, run_step: RunStep) -> None:
        self.run_id = run_step.run_id

    # thread.run.step.delta
    @override
    def on_run_step_delta(self, delta: RunStepDelta, snapshot: RunStep):
        pass

    # thread.run.completed
    @override
    def on_run_step_done(self, run_step: RunStep) -> None:
        pass

    # thread.message.created
    @override
    def on_message_created(self, message: Message) -> None:
        pass

    # thread.message.delta
    @override
    def on_message_delta(self, delta: MessageDelta, snapshot: Message) -> None:
        pass

    # thread.message.completed
    @override
    def on_message_done(self, message: Message) -> None:
        pass

    @override
    def on_tool_call_created(self, tool_call):
        pass

    @override
    def on_tool_call_delta(self, delta, snapshot):
        pass

    @override
    def on_tool_call_done(self, tool_call: ToolCall) -> None:
        if tool_call.type == 'function':
            run_status = get_run(config=self.config,
                                 params={'thread_id': self.params['thread_id'], 'run_id': self.run_id})
            while run_status['status'] in ["queued", "in_progress"]:
                run_status = get_run(config=self.config,
                                     params={'thread_id': self.params['thread_id'], 'run_id': self.run_id})
            if run_status['status'] == "requires_action":
                logger.info(
                    f'Calling tool function, Function Name: {tool_call.function.name}, Function Argument: {tool_call.function.arguments}')
                to_add_message, function_output, self.function_call_token_usage = self.call_required_function(
                    tool_call.function.name,
                    tool_call.function.arguments)
                logger.info(f'Function Calling output: {function_output}')
                logger.info(f'token_usage: {self.function_call_token_usage}')
                if to_add_message:
                    self.tool_outputs.append({"tool_call_id": tool_call.id, "output": function_output})
                else:
                    cancel_run(config=self.config,
                               params={'thread_id': self.params['thread_id'], 'run_id': self.run_id})
                    create_thread_message(self.config,
                                          params={'thread_id': self.params['thread_id'], 'role': 'assistant',
                                                  'content': function_output})

    @override
    def on_end(self):
        client = openai.OpenAI(api_key=self.config['apiKey'], project=self.config.get('project'),
                               organization=self.config.get('organization'))
        run_object = client.beta.threads.runs.retrieve(
            run_id=self.run_id,
            thread_id=self.params['thread_id']
        )
        if run_object.status == 'requires_action':
            with client.beta.threads.runs.submit_tool_outputs_stream(
                    thread_id=self.params['thread_id'],
                    run_id=self.run_id,
                    tool_outputs=self.tool_outputs,
                    event_handler=EventHandler(self.config, self.params, self.last_message_id)
            ) as stream:
                stream.until_done()
        while run_object.status not in ['completed', 'cancelled']:
            run_object = client.beta.threads.runs.retrieve(
                run_id=self.run_id,
                thread_id=self.params['thread_id']
            )
        self.llm_response = self.list_messages()
        if self.function_call_token_usage is not None:
            self.token_usage = self.set_token_usage(run_object.usage)
        else:
            self.token_usage = run_object.usage

    @override
    def on_exception(self, exception: Exception) -> None:
        logger.error(f"Exception: {exception}\n", end="", flush=True)

    def get_resposne(self):
        return self.llm_response

    def call_required_function(self, function_name, arguments):
        try:
            payload = {"connector_name": 'openai', "config_id": self.config['config_id'],
                       "function_name": function_name, "arguments": arguments,
                       "ioc_data": self.params['ioc_data'],
                       "auth_token": self.params['auth_token'],
                       "record_iri": self.params['record_iri'],
                       "record_data": self.params['record_data'],
                       "page_name": self.params['page_name']}
            response = execute_connector_action(None, 'aiassistant-utils', 'tool_function_caller', payload)
            if response.get('status') == 'Success':
                    to_add_message = response['data'].get('to_add_message')
                    llm_response = response['data'].get('data')
                    token_usage = response['data'].get('token_usage')
                    return to_add_message, llm_response, token_usage
            error_message = response.get('message')
            if not error_message:
                error_message = 'Unknown error occurred.'
            return True, error_message, {}
        except Exception as error:
            return True, f'Error occurred while executing tool function call: {error}', {}

    def set_token_usage(self, token_usage):
        self.function_call_token_usage = dict(self.function_call_token_usage)
        token_usage = dict(token_usage)

        for key in self.function_call_token_usage:
            if key in token_usage:
                token_usage[key] += self.function_call_token_usage[key]

        # Convert back to list of lists
        final_token_usage = [[key, value] for key, value in token_usage.items()]
        return final_token_usage

    def list_messages(self):
        thread_messages = list_thread_messages(config=self.config, params={'thread_id': self.params['thread_id'],
                                                                           'before': self.last_message_id})
        return [
            message['content'][0]['text']['value']
            for message in thread_messages.get('data', [])
            if message.get('content') and message['content'][0].get('type') == 'text'
        ]
