import json
from datetime import datetime
from collections import defaultdict

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

def interpreter_example(topic, message):
    print(f"Interpreter received message on {topic}: {message}")
    try:
        # Simulate interaction with an object
        obj = {"name": "the_object"}
        if "scale" in message:
            obj["scale"] = message["scale"]
            print(f"Object scale set to: {message['scale']}")
        elif "message" in message:
            obj["text"] = message["message"]
            print(f"Object text set to: {message['message']}")
    except AttributeError as e:
        print(f"Error accessing object: {e}")

def main():
    actor = "User123"
    topic = "example/publish"
    content = "This is an example content."

    # Subscribe to topic
    message_broker.subscribe(topic, interpreter_example)

    # Create and publish a subscribe activity
    subscribe_msg = subscribe_activity(actor, topic)
    message_broker.publish(topic, subscribe_msg)

    # Create and publish a note activity
    note_msg = publish_activity(actor, content, topic)
    message_broker.publish(topic, note_msg)

    # Simulate object containing subscriptions and messages
    own = {
        "subscribe_topic": "example/publish",
        "publish_message": ("example/publish", {"message": "Hello, World!"})
    }

    if "subscriber" not in own:
        own["subscriber"] = message_broker
        message_broker.subscribe("example/publish", interpreter_example)
        print("Subscriber created")

    if "subscribe_topic" in own:
        topic = own["subscribe_topic"]
        message_broker.subscribe(topic, interpreter_example)
        del own["subscribe_topic"]

    if "publish_message" in own:
        topic, content = own["publish_message"]
        message_broker.publish(topic, content)
        del own["publish_message"]

    print("Main function is running")

if __name__ == "__main__":
    main()
