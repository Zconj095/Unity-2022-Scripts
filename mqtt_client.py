import json
from collections import defaultdict, OrderedDict
from datetime import datetime

class MessageBroker:
    def __init__(self):
        self.subscriptions = defaultdict(list)

    def subscribe(self, topic, interpreter):
        if interpreter not in self.subscriptions[topic]:
            self.subscriptions[topic].append(interpreter)
            print(f"Subscribed to new topic: {topic}")

    def publish(self, topic, activity):
        if topic in self.subscriptions:
            message = json.dumps(activity)
            print(f"Published to topic: {topic}, activity: {activity}")
            for interpreter in self.subscriptions[topic]:
                interpreter(topic, json.loads(message))
        else:
            print(f"No subscribers for topic: {topic}")

# Create a global message broker instance
message_broker = MessageBroker()

def create_activitypub_message(activity_type, actor, object, target=None):
    message = {
        "@context": "https://www.w3.org/ns/activitystreams",
        "type": activity_type,
        "actor": actor,
        "object": object,
        "published": datetime.utcnow().isoformat() + "Z"
    }
    if target:
        message["target"] = target
    return message

def subscribe_activity(actor, topic):
    return create_activitypub_message("Subscribe", actor, {"type": "Topic", "name": topic})

def publish_activity(actor, content, topic):
    return create_activitypub_message("Create", actor, {"type": "Note", "content": content, "topic": topic})

# Example interpreter function
def example_interpreter(topic, payload):
    print(f"Interpreter received message on topic '{topic}': {payload}")

# Example usage
if __name__ == "__main__":
    actor = "User123"
    topic = "example/topic"
    content = "This is an example content."

    # Subscribe to topic
    message_broker.subscribe(topic, example_interpreter)

    # Create and publish a subscribe activity
    subscribe_msg = subscribe_activity(actor, topic)
    message_broker.publish(topic, subscribe_msg)

    # Create and publish a note activity
    note_msg = publish_activity(actor, content, topic)
    message_broker.publish(topic, note_msg)
