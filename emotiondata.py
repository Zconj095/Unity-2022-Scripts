class Empath:
    def __init__(self, name, energy=100, boundary=50):
        self.name = name
        self.energy = energy
        self.boundary = boundary
        self.emotional_state = "Neutral"
    
    def sense_emotion(self, emotion, intensity):
        if intensity > self.boundary:
            self.emotional_state = emotion
            self.energy -= intensity
        else:
            self.emotional_state = "Neutral"
        print(f"{self.name} senses {emotion} with intensity {intensity}. Current energy: {self.energy}.")
    
    def recharge(self, amount):
        self.energy += amount
        self.emotional_state = "Recharged"
        print(f"{self.name} recharges by {amount}. Current energy: {self.energy}.")
    
    def set_boundary(self, new_boundary):
        self.boundary = new_boundary
        print(f"{self.name}'s boundary set to {self.boundary}.")

    def interact(self, emotion, intensity):
        self.sense_emotion(emotion, intensity)
        if self.energy < 20:
            print(f"{self.name} is feeling overwhelmed and needs to recharge.")
            self.recharge(30)

# Create an empath
empath = Empath(name="Alex")

# Simulate interactions
interactions = [
    ("Sadness", 30),
    ("Joy", 20),
    ("Anger", 60),
    ("Calm", 10),
    ("Stress", 40)
]

for emotion, intensity in interactions:
    empath.interact(emotion, intensity)

# Set a new boundary and simulate more interactions
empath.set_boundary(40)

more_interactions = [
    ("Happiness", 35),
    ("Fear", 50),
    ("Love", 25)
]

for emotion, intensity in more_interactions:
    empath.interact(emotion, intensity)

class Emotion:
    def __init__(self, name, intensity, description):
        self.name = name
        self.intensity = intensity
        self.description = description

    def __str__(self):
        return f"{self.name} (Intensity: {self.intensity}): {self.description}"

class EmotionalEssence:
    def __init__(self):
        self.emotions = []

    def add_emotion(self, emotion):
        self.emotions.append(emotion)
        print(f"Added emotion: {emotion}")

    def reflect(self):
        if not self.emotions:
            print("No emotions recorded.")
            return

        print("Reflection on Emotional Essence:")
        for emotion in self.emotions:
            print(emotion)

        # Calculate and display average intensity
        avg_intensity = sum(e.intensity for e in self.emotions) / len(self.emotions)
        print(f"\nAverage Emotion Intensity: {avg_intensity:.2f}")

        # Display most common emotion
        from collections import Counter
        emotion_names = [e.name for e in self.emotions]
        common_emotion = Counter(emotion_names).most_common(1)[0][0]
        print(f"Most Common Emotion: {common_emotion}")

# Example usage
essence = EmotionalEssence()

# Adding emotions
essence.add_emotion(Emotion("Happiness", 8, "Feeling very happy today!"))
essence.add_emotion(Emotion("Sadness", 6, "Feeling a bit down."))
essence.add_emotion(Emotion("Anger", 7, "Got frustrated with work."))
essence.add_emotion(Emotion("Happiness", 9, "Had a great day with friends!"))
essence.add_emotion(Emotion("Fear", 5, "Feeling anxious about tomorrow's meeting."))

# Reflecting on emotional essence
essence.reflect()

class EmotionalState:
    def __init__(self, emotion, intensity, context):
        self.emotion = emotion
        self.intensity = intensity
        self.context = context

    def __str__(self):
        return f"{self.emotion} (Intensity: {self.intensity}): {self.context}"

class Interaction:
    def __init__(self, other_person, emotion_detected, reaction, notes):
        self.other_person = other_person
        self.emotion_detected = emotion_detected
        self.reaction = reaction
        self.notes = notes

    def __str__(self):
        return f"With {self.other_person}, detected {self.emotion_detected}. Reaction: {self.reaction}. Notes: {self.notes}"

class EmotionalPresence:
    def __init__(self):
        self.emotional_states = []
        self.interactions = []

    def add_emotional_state(self, state):
        self.emotional_states.append(state)
        print(f"Logged emotional state: {state}")

    def add_interaction(self, interaction):
        self.interactions.append(interaction)
        print(f"Logged interaction: {interaction}")

    def reflect_on_presence(self):
        if not self.emotional_states and not self.interactions:
            print("No emotional states or interactions recorded.")
            return

        print("\nReflection on Emotional Presence:")
        
        if self.emotional_states:
            print("\nEmotional States:")
            for state in self.emotional_states:
                print(state)

        if self.interactions:
            print("\nInteractions:")
            for interaction in self.interactions:
                print(interaction)
        
        # Calculate and display average emotional intensity
        if self.emotional_states:
            avg_intensity = sum(s.intensity for s in self.emotional_states) / len(self.emotional_states)
            print(f"\nAverage Emotional Intensity: {avg_intensity:.2f}")
        
        # Display most common emotion detected in interactions
        if self.interactions:
            from collections import Counter
            detected_emotions = [i.emotion_detected for i in self.interactions]
            common_emotion = Counter(detected_emotions).most_common(1)[0][0]
            print(f"Most Common Emotion Detected in Interactions: {common_emotion}")

# Example usage
presence_tracker = EmotionalPresence()

# Logging emotional states
presence_tracker.add_emotional_state(EmotionalState("Calm", 7, "Morning meditation"))
presence_tracker.add_emotional_state(EmotionalState("Stress", 8, "Project deadline"))
presence_tracker.add_emotional_state(EmotionalState("Joy", 9, "Lunch with a friend"))

# Logging interactions
presence_tracker.add_interaction(Interaction("John", "Anxiety", "Listened actively", "John seemed relieved after talking."))
presence_tracker.add_interaction(Interaction("Emily", "Happiness", "Shared a laugh", "Enjoyed the moment together."))
presence_tracker.add_interaction(Interaction("Mike", "Frustration", "Offered support", "Discussed solutions to the problem."))

# Reflecting on emotional presence
presence_tracker.reflect_on_presence()

class EmotionalTurbulence:
    def __init__(self, emotion, intensity, trigger, context):
        self.emotion = emotion
        self.intensity = intensity
        self.trigger = trigger
        self.context = context

    def __str__(self):
        return f"{self.emotion} (Intensity: {self.intensity}): Trigger - {self.trigger}, Context - {self.context}"

class Strategy:
    def __init__(self, emotion, suggestion):
        self.emotion = emotion
        self.suggestion = suggestion

    def __str__(self):
        return f"To manage {self.emotion}, try: {self.suggestion}"

class EmotionalTurbulenceTracker:
    def __init__(self):
        self.turbulences = []
        self.strategies = {
            "anger": "Practice deep breathing or take a walk to cool down.",
            "stress": "Engage in mindfulness meditation or yoga.",
            "anxiety": "Try deep breathing exercises or talk to a friend.",
            "joy": "Channel your energy into creative projects or exercise.",
            "sadness": "Write in a journal or talk to a therapist."
        }

    def add_turbulence(self, turbulence):
        self.turbulences.append(turbulence)
        print(f"Logged emotional turbulence: {turbulence}")

    def suggest_strategy(self, emotion):
        suggestion = self.strategies.get(emotion.lower(), "No specific strategy found.")
        return Strategy(emotion, suggestion)

    def reflect(self):
        if not self.turbulences:
            print("No emotional turbulence events recorded.")
            return

        print("\nReflection on Emotional Turbulence:")
        for turbulence in self.turbulences:
            print(turbulence)

        # Identify the most common trigger
        from collections import Counter
        triggers = [t.trigger for t in self.turbulences]
        common_trigger = Counter(triggers).most_common(1)[0][0]
        print(f"\nMost Common Trigger: {common_trigger}")

        # Provide strategy suggestions based on emotions
        emotions = [t.emotion for t in self.turbulences]
        for emotion in set(emotions):
            strategy = self.suggest_strategy(emotion)
            print(strategy)

# Example usage
tracker = EmotionalTurbulenceTracker()

# Logging emotional turbulence events
tracker.add_turbulence(EmotionalTurbulence("Anger", 8, "Traffic jam", "Stuck in traffic for an hour."))
tracker.add_turbulence(EmotionalTurbulence("Stress", 9, "Work deadline", "Project deadline approaching."))
tracker.add_turbulence(EmotionalTurbulence("Joy", 7, "Promotion", "Received a promotion at work."))
tracker.add_turbulence(EmotionalTurbulence("Anxiety", 6, "Public speaking", "Had to give a presentation."))
tracker.add_turbulence(EmotionalTurbulence("Sadness", 5, "Family issue", "Had an argument with a family member."))

# Reflecting on emotional turbulence
tracker.reflect()

class InnerEmotion:
    def __init__(self, emotion, intensity, trigger, context):
        self.emotion = emotion
        self.intensity = intensity
        self.trigger = trigger
        self.context = context

    def __str__(self):
        return f"Inner Emotion: {self.emotion} (Intensity: {self.intensity}) - Trigger: {self.trigger}, Context: {self.context}"

class OuterExpression:
    def __init__(self, expression, context):
        self.expression = expression
        self.context = context

    def __str__(self):
        return f"Outer Expression: {self.expression} - Context: {self.context}"

class EmotionalUnderstandingTracker:
    def __init__(self):
        self.inner_emotions = []
        self.outer_expressions = []

    def add_inner_emotion(self, emotion):
        self.inner_emotions.append(emotion)
        print(f"Logged inner emotion: {emotion}")

    def add_outer_expression(self, expression):
        self.outer_expressions.append(expression)
        print(f"Logged outer expression: {expression}")

    def reflect_on_emotions(self):
        if not self.inner_emotions and not self.outer_expressions:
            print("No emotions or expressions recorded.")
            return

        print("\nReflection on Emotional Understanding:")
        
        if self.inner_emotions:
            print("\nInner Emotions:")
            for emotion in self.inner_emotions:
                print(emotion)

        if self.outer_expressions:
            print("\nOuter Expressions:")
            for expression in self.outer_expressions:
                print(expression)
        
        # Identify connections between inner emotions and outer expressions
        connections = []
        for emotion in self.inner_emotions:
            for expression in self.outer_expressions:
                if emotion.context == expression.context:
                    connections.append((emotion, expression))
        
        if connections:
            print("\nConnections between Inner Emotions and Outer Expressions:")
            for connection in connections:
                print(f"{connection[0]} -> {connection[1]}")
        else:
            print("No direct connections found between inner emotions and outer expressions.")

# Example usage
tracker = EmotionalUnderstandingTracker()

# Logging inner emotions
tracker.add_inner_emotion(InnerEmotion("Anger", 8, "Traffic jam", "Stuck in traffic for an hour."))
tracker.add_inner_emotion(InnerEmotion("Stress", 9, "Work deadline", "Project deadline approaching."))
tracker.add_inner_emotion(InnerEmotion("Joy", 7, "Promotion", "Received a promotion at work."))
tracker.add_inner_emotion(InnerEmotion("Anxiety", 6, "Public speaking", "Had to give a presentation."))
tracker.add_inner_emotion(InnerEmotion("Sadness", 5, "Family issue", "Had an argument with a family member."))

# Logging outer expressions
tracker.add_outer_expression(OuterExpression("Scowling", "Stuck in traffic for an hour."))
tracker.add_outer_expression(OuterExpression("Sighing frequently", "Project deadline approaching."))
tracker.add_outer_expression(OuterExpression("Smiling", "Received a promotion at work."))
tracker.add_outer_expression(OuterExpression("Fidgeting", "Had to give a presentation."))
tracker.add_outer_expression(OuterExpression("Crying", "Had an argument with a family member."))

# Reflecting on emotional understanding
tracker.reflect_on_emotions()

class MetabolicState:
    def __init__(self, weight, sleep_hours, diet_quality, exercise_minutes, context):
        self.weight = weight
        self.sleep_hours = sleep_hours
        self.diet_quality = diet_quality
        self.exercise_minutes = exercise_minutes
        self.context = context

    def __str__(self):
        return (f"Weight: {self.weight} kg, Sleep: {self.sleep_hours} hours, "
                f"Diet Quality: {self.diet_quality}/10, Exercise: {self.exercise_minutes} minutes - Context: {self.context}")

class EmotionalState:
    def __init__(self, mood, intensity, trigger, context):
        self.mood = mood
        self.intensity = intensity
        self.trigger = trigger
        self.context = context

    def __str__(self):
        return f"Mood: {self.mood} (Intensity: {self.intensity}) - Trigger: {self.trigger}, Context: {self.context}"

class HealthTracker:
    def __init__(self):
        self.metabolic_states = []
        self.emotional_states = []

    def add_metabolic_state(self, state):
        self.metabolic_states.append(state)
        print(f"Logged metabolic state: {state}")

    def add_emotional_state(self, state):
        self.emotional_states.append(state)
        print(f"Logged emotional state: {state}")

    def analyze_health(self):
        if not self.metabolic_states and not self.emotional_states:
            print("No health data recorded.")
            return

        print("\nHealth Analysis:")

        if self.metabolic_states:
            print("\nMetabolic States:")
            for state in self.metabolic_states:
                print(state)

        if self.emotional_states:
            print("\nEmotional States:")
            for state in self.emotional_states:
                print(state)

        # Calculate and display average sleep and exercise
        if self.metabolic_states:
            avg_sleep = sum(s.sleep_hours for s in self.metabolic_states) / len(self.metabolic_states)
            avg_exercise = sum(s.exercise_minutes for s in self.metabolic_states) / len(self.metabolic_states)
            print(f"\nAverage Sleep: {avg_sleep:.2f} hours")
            print(f"Average Exercise: {avg_exercise:.2f} minutes")

        # Identify the most common triggers for emotional turbulence
        from collections import Counter
        if self.emotional_states:
            triggers = [e.trigger for e in self.emotional_states]
            common_trigger = Counter(triggers).most_common(1)[0][0]
            print(f"\nMost Common Trigger for Emotional Turbulence: {common_trigger}")

        # Provide suggestions for maintaining balance
        print("\nSuggestions for Maintaining Balance:")
        print("1. Identify and manage your triggers.")
        print("2. Practice relaxation techniques such as deep breathing, meditation, and yoga.")
        print("3. Ensure you get 7-8 hours of sleep each night.")
        print("4. Eat a balanced diet rich in fruits, vegetables, and whole grains.")
        print("5. Engage in at least 30 minutes of moderate exercise most days of the week.")
        print("6. Talk to a therapist or counselor if you need help managing emotional turbulence.")

# Example usage
tracker = HealthTracker()

# Logging metabolic states
tracker.add_metabolic_state(MetabolicState(70, 7, 8, 30, "Normal day"))
tracker.add_metabolic_state(MetabolicState(72, 6, 7, 20, "Stressful day at work"))
tracker.add_metabolic_state(MetabolicState(69, 8, 9, 40, "Relaxed weekend"))

# Logging emotional states
tracker.add_emotional_state(EmotionalState("Stress", 7, "Work deadline", "Project deadline approaching."))
tracker.add_emotional_state(EmotionalState("Happiness", 8, "Promotion", "Received a promotion at work."))
tracker.add_emotional_state(EmotionalState("Anxiety", 6, "Public speaking", "Had to give a presentation."))

# Analyzing health data
tracker.analyze_health()
