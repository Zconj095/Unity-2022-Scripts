# Simulated JSON data as a string
json_data = """
[
    {"term": "AI", "meaning": " Artificial intelligence (AI) refers to the simulation of human intelligence in machines."},
    {"term": "Machine Learning", "meaning": " Machine learning is a branch of AI focused on building applications that learn from data and improve their accuracy over time without being programmed to do so."}
]
"""

# Simulate reading JSON from a file by converting the string to a dictionary
def simulate_json_loads(json_str):
    data = []
    # Removing newlines and extra spaces
    json_str = json_str.strip().replace('\n', '').replace('  ', ' ')
    # Remove the opening and closing brackets
    json_str = json_str[1:-1]
    # Split into separate objects
    entries = json_str.split('},')
    for entry in entries:
        entry = entry.strip().rstrip(',')
        if not entry.endswith('}'):
            entry += '}'
        # Extract term and meaning
        term_start = entry.find('"term": "') + 9
        term_end = entry.find('",', term_start)
        term = entry[term_start:term_end]
        meaning_start = entry.find('"meaning": "') + 12
        meaning_end = entry.rfind('"')
        meaning = entry[meaning_start:meaning_end]
        # Append to data
        data.append({'term': term, 'meaning': meaning})
    return data

# Load definitions
definitions = simulate_json_loads(json_data)

# Dictionary to hold definitions
definition_dict = {}

# Loop through definitions and add to dictionary
for definition in definitions:
    term = definition['term']
    meaning = definition['meaning']
    
    # Clean up text
    meaning = meaning.strip()
    
    # Add to dictionary
    definition_dict[term] = meaning

# Function to get definition of term  
def define(term):
    if term in definition_dict:
        return definition_dict[term]
    else:
        return "Term not defined."

# Test    
print(define("AI"))  # Should return the definition of AI
print(define("Machine Learning"))  # Should return the definition of Machine Learning

import pickle

class MemoryModule:

    def __init__(self):
        self.memory_db = {}
        self.memory_index = {}

    def define(self, key, value):
        # Add new key-value pair to memory
        self.memory_db[key] = value

        # Extract keywords for indexing
        keywords = self._extract_keywords(value)

        # Update indexes
        for kw in keywords:
            if kw not in self.memory_index:
                self.memory_index[kw] = []
            self.memory_index[kw].append(key)

        print(f"Defined: {key} -> {value}")

    def describe(self, key):
        if key in self.memory_db:
            return self.memory_db[key]
        else:
            return f"No definition found for {key}"

    def search(self, keyword):
        if keyword in self.memory_index:
            memories = self.memory_index[keyword]
            print(f"Related memories for '{keyword}':")
            for mem in memories:
                print(f"- {mem}: {self.describe(mem)}")
        else: 
            print(f"No memories related to '{keyword}' found.")

    def _extract_keywords(self, text):
        # Simple keyword extraction
        return set(text.split())

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

mem = MemoryModule()
mem.define("apple", "a sweet fruit that grows on trees") 
mem.define("tree", "a tall plant with branches and leaves")
mem.search("fruit")
mem.save("memories.pkl")

loaded_mem = MemoryModule.load("memories.pkl")
loaded_mem.search("tree")

class MemoryModule:
    def __init__(self):
        self.memory_index = {
            # Example data, keyword mapped to some texts
            "fruit": ["Fruit is a part of plant that is often eaten", "Banana is a yellow fruit"]
        }
        # Predefined simple stopwords list
        self.stopwords = {"is", "a", "the", "and", "or", "of", "to", "in"}
    
    def _tokenize(self, text):
        # Basic tokenizer without punctuation handling
        return text.lower().split()
    
    def _remove_stopwords(self, words):
        # Filter out stopwords from the tokenized word list
        return [word for word in words if word not in self.stopwords]
    
    def _lemmatize(self, word):
        # Very simplified lemmatizer for demonstration (hardcoded for common cases)
        lemmas = {"fruits": "fruit", "apples": "apple", "bananas": "banana"}
        return lemmas.get(word, word)
    
    def _extract_keywords(self, text):
        # Tokenize text into words
        words = self._tokenize(text)
        
        # Remove stop words
        words = self._remove_stopwords(words)
        
        # Lemmatize words to base form
        keywords = [self._lemmatize(w) for w in words]
        
        # Remove duplicates
        keywords = list(set(keywords))

        return keywords

    def search(self, query):
        # Tokenize and process search query 
        query_words = self._extract_keywords(query)
        
        # Search for query keywords
        results = []
        for kw in query_words:
            if kw in self.memory_index:
                memories = [(desc, key) for key in self.memory_index[kw] for desc in self.memory_index[kw]]
                results.extend(memories)
        
        # Simplified result ranking by counting keyword matches (instead of edit distance)
        results.sort(key=lambda x: -sum(query_word in x[0] for query_word in query_words))
        
        # Return top 5 closest matches
        return results[:5]
        
# Instantiate and use the MemoryModule
mem = MemoryModule()
result = mem.search("what is a fruit")
for description, key in result:
    print(f"{description} ({key})")


import random
import time
class MemoryModule:
    def __init__(self):
        self.memories = {}
        self.memory_vectors = {}  # Initialize memory_vectors to store TF-IDF vectors

    def _vectorize(self, text):
        # Simplified version of TF-IDF vectorization
        words = text.lower().split()
        word_count = len(words)
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        # Normalize by the total number of words
        return {word: freq / word_count for word, freq in word_freq.items()}

    def define(self, key, value):
        # Vectorize memory value
        vector = self._vectorize(value)
        self.memory_vectors[key] = vector
        self.memories[key] = value
        print(f"Defined {key}: {value}")

    def search_vector(self, query):
        # Vectorize query
        query_vector = self._vectorize(query)

        # Calculate similarity by comparing vectors (cosine similarity approximation)
        def cosine_similarity(vec1, vec2):
            # Simplified cosine similarity (dot product of vectors)
            intersection = set(vec1.keys()) & set(vec2.keys())
            numerator = sum([vec1[x] * vec2[x] for x in intersection])
            sum1 = sum([val**2 for val in vec1.values()])
            sum2 = sum([val**2 for val in vec2.values()])
            denominator = (sum1**0.5) * (sum2**0.5)
            if not denominator:
                return 0.0
            else:
                return numerator / denominator

        similarities = []
        for key, vector in self.memory_vectors.items():
            similarity = cosine_similarity(query_vector, vector)
            similarities.append((key, similarity))

        # Sort by similarity and return
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [print(f"{self.memories[key]} ({key}) - {sim:.2f}") for key, sim in similarities[:5]]

# Usage:

mem = MemoryModule()
mem.define("apple", "sweet red fruit")
mem.define("banana", "long yellow fruit")

mem.search_vector("what are fruits")


class MemoryModule:
    def __init__(self):
        self.documents = {}
        self.categories = {}

    def define(self, key, value, categories=None):
        self.documents[key] = {'text': value, 'categories': categories or {}}
        print(f"Defined {key}: {value}")

    def search(self, query):
        print("Related memories:")
        query = query.lower()
        results = []
        for key, doc in self.documents.items():
            doc_text = doc['text'].lower()
            # Simple search based on text inclusion
            if query in doc_text:
                results.append(doc['text'])
        # Print results
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc}")

    def customize(self, category, texts):
        for text, is_category in texts:
            self.categories[text] = {category: is_category}
            if is_category:
                # Define the document with categories if True
                self.define(text, text, {category: True})
        print(f"Customized category '{category}' with provided texts.")

# Instantiate and use the MemoryModule
mem = MemoryModule()
fruits = [
    ("apples are sweet and juicy", True),
    ("bananas have yellow peel", True), 
    ("coffee brews from beans", False)
]

mem.customize("fruit", fruits)
mem.define("mango", "sweet oval fruit", {"fruit": True}) 
mem.search("which fruits")


class Conversation:
    def __init__(self):
        self.memory = MemoryModule()
        self.context = []

    def receive(self, input):
        # Append user input to the context
        self.context.append({"text": input, "sender": "user"})
        return self.respond(input)

    def respond(self, input):
        # Process input with memory module
        result = self.memory.process(input, self.context)

        # Add system response to context
        response = {"text": result, "sender": "system"}
        self.context.append(response)

        return result

class MemoryModule:
    def __init__(self):
        self.memory = {}

    def process(self, input, context):
        # Simplified example of processing input
        response = "Processing: " + input
        return response

    def manual_recall(self, key):
        print(f"Initiating manual recall for {key}...")

        # Simulate delay of manual recall
        delay_time = 2  # seconds
        start_time = time.time()
        while time.time() - start_time < delay_time:
            pass  # Simulating time delay without using time.sleep()

        # Verify if key is defined
        if key in self.memory:
            memory = self.memory[key]
            # Probability of accurate recall (simplified without numpy)
            from random import random
            accuracy = random()
            if accuracy > 0.8:
                return memory
            else:
                return None

        else:
            return None

    def retrieve(self, key):
        memory = self.manual_recall(key)
        if memory:
            print(f"Recalled: {memory}")
        else:
            print(f"Failed recall for {key}")

# Usage:

convo = Conversation()
print(convo.receive("Where do apples grow?"))
print(convo.receive("What color are bananas?"))

mem = MemoryModule()
mem.memory["my_password"] = "Xy*32jjk"

mem.retrieve("my_password")
mem.retrieve("phone_number")


import random

class MemoryModule:
    def __init__(self):
        self.memories = {}

    def auto_recall(self, key):
        return self.memories.get(key, "No memory defined for this key.")

    def manual_recall(self, key):
        start = random.random() * 5  # Simulate processing time
        if key in self.memories:
            recalled_memory = self._search_associations(key)
            accuracy = self._assess_accuracy(recalled_memory, key)
            print(f"Recalled: {recalled_memory} ({key}) in {start:.2f} secs with {accuracy:.2%} accuracy")
            return recalled_memory
        else:
            print(f"No memory defined for {key}")
            return None

    def _search_associations(self, key):
        # Simulated function to find associated memories
        return self.memories.get(key, "No associated memory found")

    def _assess_accuracy(self, recalled, key):
        original = self.memories.get(key, "")
        distance = len(set(recalled) ^ set(original))  # Basic measure of difference
        return 1 - (distance / max(len(recalled), len(original)))

class Memory:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.mutations = []

    def mutate(self, subjection):
        mutant = self._apply_subjection(subjection)
        self.mutations.append(mutant)

    def recall(self):
        print(f"Recalling: {self.value}")
        for m in self.mutations:
            print(f"Mutation: {m}")

    def _apply_subjection(self, sub):
        # Example mutation logic
        return sub + " modified"

# Example usage
mem_module = MemoryModule()
mem_module.memories['friend'] = "Someone you trust"
print(mem_module.auto_recall('friend'))
mem_module.manual_recall('friend')

memory1 = Memory("apple", "red fruit")
memory1.mutate("apples are green before ripe")
memory1.recall()


import random
import uuid
class MemoryNetwork:

    def __init__(self): 
        self.memories = {}
        self.contexts = []

    def perceive(self, event):
        # Add perceived event to memories
        key = uuid.uuid4()
        self.memories[key] = event

    def context(self, event):
        # Add contextual event   
        self.contexts.append(event)
        
    def auto_respond(self):
        # Probability of automatic response  
        if random.random() < 0.2:  
            memory = self.recall_related(self.contexts[-1])
            if memory is not None:
                print(f"Memory retrieved: {memory}")
               
    def recall_related(self, context):
        # Logic to selectively recall related memories  
        for key, memory in self.memories.items():
            if is_related(context, memory):
                return memory
        
        return None 

def is_related(context, memory):
    # Custom logic 
    pass
    
net = MemoryNetwork()

# Add memories...
net.perceive("Went to the beach") 
net.context("Looking at waves")
net.auto_respond()

import uuid
from datetime import datetime

class MicromanagedMemory:
    def __init__(self):
        self.memories = {}

    def perceive(self, event):
        # Store event with micro details
        key = uuid.uuid4()  # Generate a unique key for the event
        timestamp = datetime.now()
        location = self._get_location()
        emotional_state = self._get_emotion()
        sensory_details = self._get_senses()

        memory = {
            "event": event,
            "timestamp": timestamp,
            "location": location,
            "emotional_state": emotional_state,
            "sensory_details": sensory_details
        }

        self.memories[key] = memory
        return key  # Return the key so it can be used for recalling the memory

    def recall(self, key):
        memory = self.memories.get(key)
        if memory:
            print("Recalling memory:")
            print(f"- Event: {memory['event']}")
            print(f"- Timestamp: {memory['timestamp']}")
            print(f"- Location: {memory['location']}")
            print(f"- Emotional state: {memory['emotional_state']}")
            print(f"- Sensory details: {memory['sensory_details']}")
        else:
            print("Memory not found.")

    def _get_location(self):
        # Logic to track precise location
        return "Unknown location"  # Placeholder for location logic

    def _get_emotion(self):
        # Logic to detect emotion at encoding time
        return "Neutral"  # Placeholder for emotion detection logic

    def _get_senses(self):
        # Logic to capture sensory snapshot
        return "General sensory details"  # Placeholder for sensory details logic

# Example usage

mem = MicromanagedMemory()

event_key = mem.perceive("I went to the store")
mem.recall(event_key)  # Use the returned key to recall the memory


class MemoryStore:
    def __init__(self):
        self.memories = {}

    def store(self, key, event):
        if not callable(self.store):
            raise Exception("MemoryStore.store has been overwritten")
        self.memories[key] = event

    def retrieve(self, key):
        if not callable(self.retrieve):
            raise Exception("MemoryStore.retrieve has been overwritten")
        return self.memories.get(key, "Memory not found.")

class MemoryIndexer:
    def __init__(self):
        self.index = {}

    def index(self, key, event):
        if not callable(self.index):
            raise Exception("MemoryIndexer.index has been overwritten")
        words = event.split()
        for word in words:
            if word not in self.index:
                self.index[word] = []
            self.index[word].append(key)

    def lookup(self, trigger):
        if not callable(self.lookup):
            raise Exception("MemoryIndexer.lookup has been overwritten")
        return self.index.get(trigger, [])

class MemoryRecollection:
    def __init__(self, memory_module, indexer_module):
        self.memory = memory_module
        self.indexer = indexer_module
        self.next_id = 0

    def perceive(self, event):
        print(f"Before calling store or index: store type={type(self.memory.store)}, index type={type(self.indexer.index)}")
        key = self.next_id
        self.next_id += 1
        self.memory.store(key, event)
        self.indexer.index(key, event)
        print(f"After calling store and index: store type={type(self.memory.store)}, index type={type(self.indexer.index)}")


# Initialize the memory storage and indexing system
memory_store = MemoryStore()
memory_indexer = MemoryIndexer()
diary = MemoryRecollection(memory_store, memory_indexer)





import uuid

class MemoryPipeline:

    def __init__(self):
        self.memories = {}
        self.index = {}

    def absorb(self, event, categories):
        # Store memory
        key = uuid.uuid4()  
        self.memories[key] = Memory(event)

        # Index in categories
        for c in categories:
            if c not in self.index:
                self.index[c] = []
            self.index[c].append(key)

    def recall(self, category):
        if category in self.index:
            memory_keys = self.index[category]
            memories = [self.memories[key] for key in memory_keys] 
            return memories
        else:
            return []
            
class Memory:

    def __init__(self, event):  
        self.event = event

pipe = MemoryPipeline()
pipe.absorb("Went to beach", ["vacation", "beach"])
pipe.absorb("Meeting with team", ["work"])
pipe.recall("vacation")

import datetime

class MemoryRelay:

    def __init__(self, memory):
        self.memory = memory
        self.short_term = []

    def perceive(self, event):
        # Store event
        key = uuid.uuid4()  
        self.memory.store(key, event)

        # Add to short term  
        exp_time = datetime.datetime.now() + datetime.timedelta(minutes=5)  
        entry = (key, exp_time)
        self.short_term.append(entry)

    def pre_respond(self, event):
        self._relay(event, pre=True)

    def post_respond(self, event):
        self._relay(event)

    def _relay(self, event, pre=False):
        if self.short_term: 
            responses = []
            # Check short term memories
            for key, exp in self.short_term:
                if datetime.datetime.now() < exp:  
                    memory = self.memory.retrieve(key)
                    if self._is_related(event, memory):
                        response = self._format_response(memory, pre=pre)
                        responses.append(response)
                        
            # Return relayed responses                
            return responses            
    
    # Helper methods
    def _is_related(self, event, memory):
        pass

    def _format_response(self, memory, pre=False):
        pass
    
relay = MemoryRelay(memory_store)

import uuid  # Make sure to import uuid for unique keys
import numpy as np

class EnhancingMemory:
    def __init__(self, decay=0.5): 
        self.memories = {}  # Store {key: (event, strength)}
        self.decay = decay

    def perceive(self, event):
        # Store event with a unique key and initial strength
        key = uuid.uuid4()   
        self.memories[key] = (event, 1.0)  # Store event with initial strength

    def recall(self, event):
        keys = self._retrieve_keys(event)
        if keys:
            # Strengthen matching memories
            for key in keys:
                event_description, strength = self.memories[key]
                self.memories[key] = (event_description, strength + 0.1)  # Strengthen memory
            print(f"Memory reactivated for similar event: {event}")
        else:
            print(f"New event perceived: {event}")

    def _retrieve_keys(self, event):
        # Logic to find related memory keys
        return [key for key, (stored_event, strength) in self.memories.items() if event in stored_event]

    def update(self): 
        # Apply temporal decay to all memories
        for key in list(self.memories.keys()):
            event_description, strength = self.memories[key]
            new_strength = strength * self.decay
            # Update memory strength
            self.memories[key] = (event_description, new_strength)
            # Optionally remove memories that fall below a certain threshold
            if new_strength < 0.1:
                del self.memories[key]  # Cleanup low-strength memories

# Example event stream
event_stream = [
    "Went to the beach for a relaxing day.",
    "Celebrated my birthday with friends.",
    "Worked on a project at home.",
    "Visited the local beach.",
    "Had a fun beach day with family."
]

# Create instance of EnhancingMemory
mem = EnhancingMemory() 

# Process stream of events
for event in event_stream:
    mem.perceive(event)  
    mem.recall(event)    
    mem.update()  # Decay cycle


import uuid  # Ensure to import uuid for generating unique keys

class Memory:
    def __init__(self, key, content):
        self.key = key
        self.content = content

class Intuition:
    def __init__(self, cue, related):
        self.cue = cue
        self.related_categories = related

class IntuitiveMemory:
    def __init__(self):
        self.categories = {}  # Maps category to list of memory keys
        self.intuitions = {}  # Maps intuition keys to Intuition objects

    def absorb(self, memory_content, categories):
        # Store memory
        key = uuid.uuid4()  
        mem = Memory(key, memory_content)
        
        # Store the memory under the unique key
        self.categories[key] = mem

        # Index categories
        for cat in categories:
            if cat not in self.categories:
                self.categories[cat] = []
            self.categories[cat].append(key)

    def intuit(self, cue, related):
        # Store intuition  
        key = uuid.uuid4()
        self.intuitions[key] = Intuition(cue, related)

    def recall(self, cue):
        # Lookup intuition 
        intuition = self._find_intuition(cue)
        if intuition is None:
            return []

        # Retrieve related categories
        categories = intuition.related_categories 
        memory_keys = []
        for cat in categories:
            if cat in self.categories:
                memory_keys.extend(self.categories[cat]) 

        memories = [self.categories[k].content for k in memory_keys]  # Retrieve the content of memories
        return memories

    def _find_intuition(self, cue):
        # Find matching intuition (if any)
        for key, intuition in self.intuitions.items():
            if intuition.cue == cue:
                return intuition
        return None

# Example usage
intuitive_memory = IntuitiveMemory()

# Absorb memories with categories
intuitive_memory.absorb("Enjoyed a sunny day at the park.", ["outdoors", "leisure"])
intuitive_memory.absorb("Learned about Python programming.", ["education", "work"])
intuitive_memory.absorb("Had a great birthday party with friends.", ["celebration", "friends"])

# Store an intuition
intuitive_memory.intuit("celebration", ["celebration", "friends"])

# Recall memories related to "celebration"
celebration_memories = intuitive_memory.recall("celebration")
print("Memories related to 'celebration':")
for memory in celebration_memories:
    print(memory)  # Output the contents of related memories


import numpy as np
import uuid  # Ensure uuid is imported for generating unique keys

class MemoryStore:
    def __init__(self):
        self.memories = {}

    def store(self, key, event):
        self.memories[key] = event

    def retrieve(self, key):
        return self.memories.get(key, None)  # Return None if memory not found

class RecallEnhancer:
    def __init__(self, memory, decay=0.5):
        self.memory = memory
        self.success = {}
        self.decay = decay

    def recall(self, cue):
        memory = self.memory.retrieve(cue)  # Use the cue to retrieve memory
        if memory is None:
            return None

        self.success[cue] = self.success.get(cue, 1.0)  # Initialize success if not present
        return memory

    def enhance(self):
        for cue in list(self.success):
            if np.random.random() < self.success[cue]:
                # Recall success - strengthen
                print(f"Strengthening recall for {cue}")
                self.success[cue] += 0.1
            else:
                # Recall failure - weaken
                print(f"Weakening recall for {cue}")
                self.success[cue] -= 0.05

        # Apply decay
        for cue in list(self.success):
            self.success[cue] *= self.decay

# Example usage

# Initialize the memory store
memory_store = MemoryStore()

# Store some memories
memory_store.store("my_memory", "This is an important memory.")

# Initialize the RecallEnhancer with the correct memory instance
recaller = RecallEnhancer(memory_store)

# Recall a specific memory
memory_item = recaller.recall("my_memory")
print(f"Recalled memory: {memory_item}")  # Output: "This is an important memory."

# Periodic enhancement step
recaller.enhance()


import uuid  # Make sure to import uuid for generating unique keys
import numpy as np

class MemoryStore:
    def __init__(self):
        self.memories = {}

    def store(self, key, event):
        self.memories[key] = event

    def retrieve(self, key):
        return self.memories.get(key, "Memory not found.")

class MemoryEnhancer:
    def __init__(self, memory, loss_rate=0.1):
        self.memory = memory
        self.strengths = {}
        self.loss_rate = loss_rate

    def perceive(self, event):
        key = uuid.uuid4()  # Generate a unique key
        self.memory.store(key, event)  # Store the event with the key
        self.strengths[key] = 1.0  # Initialize strength for this key

    def access(self, key):
        if key in self.strengths:  # Check if the key exists in strengths
            self.strengths[key] += 0.5  # Increase strength
            self.memory.retrieve(key)  # Retrieve the memory associated with the key
        else:
            print(f"Key {key} not found in strengths.")

    def process(self):
        # Decay strengths  
        for key in list(self.strengths.keys()):
            self.strengths[key] -= self.strengths[key] * self.loss_rate
            self.strengths[key] = max(0, self.strengths[key])  # Prevent negative strengths

    def infer(self, event):
        # Reinforce related memories  
        keys = self._retrieve_related(event)  
        if keys:
            for key in keys:
                self.strengths[key] += 0.25  # Increase strength for related memories

    def _retrieve_related(self, event):
        # Logic to retrieve related keys (not implemented in this example)
        # For now, return all keys for simplicity
        return list(self.strengths.keys())

# Example event stream
event_stream = [
    "Went to the beach for a relaxing day.",
    "Celebrated my birthday with friends.",
    "Worked on a project at home."
]

# Initialize the MemoryStore and MemoryEnhancer
memory_store = MemoryStore()
mem = MemoryEnhancer(memory_store)

# Process stream of experiences 
for event in event_stream:
    mem.perceive(event)  # Store the event

# Access memories using the stored keys
for key in list(mem.strengths.keys()):
    mem.access(key)  # Access the memory to strengthen it
    mem.infer(event)  # Call infer for each event
    mem.process()  # Decay step

# Check current strengths
print("Current strengths:", mem.strengths)


import uuid  # Ensure to import uuid for generating unique keys

class InstantMemory:
    def __init__(self):
        self.groups = {}  # Dictionary to hold memories categorized by groups
        self.filters = {}  # Dictionary to hold sets of keys for each group

    def absorb(self, memory, groups):
        # Store memory with a unique key
        key = uuid.uuid4()  # Generate a unique identifier for the memory
        self.groups[key] = memory  # Store the memory

        # Add the key to the appropriate group and update the filter
        for g in groups:
            if g not in self.filters:
                self.filters[g] = set()  # Initialize a set for the group
            self.filters[g].add(key)  # Add the memory key to the group set

    def recall(self, group):
        if group in self.filters:
            keys = self.filters[group]  # Get the keys for the group
            memories = [self.groups[k] for k in keys]  # Retrieve memories using the keys
            print(f"Instantly recalled {len(memories)} memories from {group}")
            return memories
        else:
            print(f"No memories found for group: {group}")
            return []

# Example Usage
mem = InstantMemory()
mem.absorb("Beach trip", ["vacations", "beaches"])
mem.absorb("Meeting notes", ["work"])

# Recall memories from the "vacations" group
recalled_memories = mem.recall("vacations")
print(recalled_memories)

# Instantly recalled 1 memories from vacations

#______________________________________________________________#
