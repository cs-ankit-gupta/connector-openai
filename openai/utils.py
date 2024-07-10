from integrations.crudhub import make_request


def execute_connector_action(config_id, connector_name, operation, payload):
    url = '/api/integration/execute/?format=json'
    method = 'POST'
    payload = {
        "connector": connector_name,
        "version": "3.0.0",
        "config": config_id,
        "operation": operation,
        "params": payload,
        "audit": False
    }
    return make_request(url, method, body=payload)
