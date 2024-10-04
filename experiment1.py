system_interface_values = 42.826
hyperflux_delay =.000002
divisional_cortex_value = 44.44
dynamic_rate_of_change = 22 * 1/0.11872
Overall_Dynamics = system_interface_values * system_interface_values/0.22815 * 32 + 32
synchronous_systematic_feedback = 0.777777777 * dynamic_rate_of_change/hyperflux_delay
intermean_values = synchronous_systematic_feedback * system_interface_values/Overall_Dynamics
dynamic_movement = synchronous_systematic_feedback * intermean_values*(dynamic_rate_of_change/2)
subjective_mean = 52
cortical_region_strucuture = subjective_mean/system_interface_values
frequency_overlay = .223785
intersect_values = .47
overall_mean = frequency_overlay*(Overall_Dynamics/Overall_Dynamics)*subjective_mean/frequency_overlay*(intersect_values*system_interface_values/synchronous_systematic_feedback)
print(system_interface_values * hyperflux_delay/divisional_cortex_value)
print(synchronous_systematic_feedback * cortical_region_strucuture)

from dataclasses import dataclass
from enum import Enum
from typing import List
import uuid

class StatType(Enum):
    STRENGTH = "Strength"
    POWER = "Power" 
    AGILITY = "Agility"
    
    
@dataclass  
class Stat:
    type: StatType
    base_value: int

    
@dataclass   
class Character:
    id: uuid.UUID
    name: str
    stats: List[Stat]
    
    level: int = 1
    experience: int = 0
    
    def add_experience(self, amt):
        self.experience += amt
        if self.experience >= self.get_xp_for_level():
            self.level_up()
            
    def get_xp_for_level(self):
        return 100 * (self.level ** 2)
    
    def level_up(self):
        print(f"{self.name} leveled up to {self.level+1}!")
        self.level += 1
        

maya = Character(
    id = uuid.uuid4(), 
    name = "Maya",
    stats = [
        Stat(StatType.STRENGTH, 28),
        Stat(StatType.POWER, 24),
        Stat(StatType.AGILITY, 24)
    ]
)

print(maya.name, maya.level, maya.stats) 

maya.add_experience(500)

from dataclasses import dataclass
from typing import List

@dataclass
class Skill:
    name: str
    level: int = 1
    
    def level_up(self):
        self.level += 1
        
@dataclass        
class Character:
    # Existing fields
    
    skills: List[Skill] = None
    
    def learn_skill(self, skill):
        if self.skills is None:
            self.skills = []
        self.skills.append(skill)
        
    def train_skill(self, skill):
        skill.level_up()
        
# Usage
fireball = Skill("Fireball") 
maya = Character()

maya.learn_skill(fireball)  
maya.train_skill(fireball)

print(maya.skills[0].name, maya.skills[0].level)
# Fireball 2

from dataclasses import dataclass
from typing import List

@dataclass
class Skill:
    name: str 
    cost: int
    level: int = 1
    
    def level_up(self):
        self.level += 1


@dataclass   
class Character:
    skills: List[Skill]
    skill_points: int = 5
    
    def learn_skill(self, skill):
        if self.skill_points >= skill.cost:
            self.skills.append(skill)   
            self.skill_points -= skill.cost
            
    def train_skill(self, skill):
        if skill in self.skills:
            skill.level += 1
            

fireball = Skill("Fireball", 3)
frostbolt = Skill("Frostbolt", 2)
            
from dataclasses import dataclass, field 
from typing import List

@dataclass
class Skill:
    name: str
    cost: int 
    
@dataclass
class Character:
    skills: List[Skill] = field(default_factory=list)
    
    def learn_skill(self, skill):
        self.skills.append(skill)

fireball = Skill("Fireball", 3)

maya = Character()
maya.learn_skill(fireball) 

print(maya.skills)

@dataclass
class Skill:
    name: str
    cost: int
    level: int = 1
    
    def train(self):
        self.level += 1
        
@dataclass        
class Character:
    skills: List[Skill] = field(default_factory=list)
    skill_points: int = 5
    
    def learn_skill(self, skill):
        if self.skill_points >= skill.cost:
            self.skills.append(skill)
            self.skill_points -= skill.cost
            
    def train_skill(self, skill):
        if skill in self.skills:
            skill.train()

fireball = Skill("Fireball", 3)
maya = Character()

maya.learn_skill(fireball) 
maya.train_skill(fireball)

print(fireball.level) # Level 2

from random import randint

@dataclass 
class Character:
    strength: int = 10
    
    def attack(self):
        dmg = self.strength + randint(0, 5)
        print(f"You attack for {dmg} damage!")
        
@dataclass        
class Enemy:
    name: str
    health: int = 100
    
    def take_damage(self, dmg):
        self.health -= dmg 
        print(f"{self.name} takes {dmg} damage!")
        
maya = Character()  
orc = Enemy(name="Orc")

maya.attack()  
orc.take_damage(15)   

while orc.health > 0:
    maya.attack()
    orc.take_damage(randint(10, 15))
    
print("You defeated the orc!")