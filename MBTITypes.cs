using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MBTITypes : MonoBehaviour
{
    // Structure to represent a personality type
    [System.Serializable]
    public struct PersonalityType
    {
        public string name;
        public string code;
    }

    // List to store all personality types
    public List<PersonalityType> personalityTypes = new List<PersonalityType>
    {
        new PersonalityType { name = "ISTJ", code = "The Inspector" },
        new PersonalityType { name = "ISTP", code = "The Crafter" },
        new PersonalityType { name = "ISFJ", code = "The Protector" },
        new PersonalityType { name = "ISFP", code = "The Artist" },
        new PersonalityType { name = "INFJ", code = "The Advocate" },
        new PersonalityType { name = "INFP", code = "The Mediator" },
        new PersonalityType { name = "INTJ", code = "The Architect" },
        new PersonalityType { name = "INTP", code = "The Thinker" },
        new PersonalityType { name = "ESTP", code = "The Persuader" },
        new PersonalityType { name = "ESTJ", code = "The Director" },
        new PersonalityType { name = "ESFP", code = "The Performer" },
        new PersonalityType { name = "ESFJ", code = "The Caregiver" },
        new PersonalityType { name = "ENFP", code = "The Champion" },
        new PersonalityType { name = "ENFJ", code = "The Giver" },
        new PersonalityType { name = "ENTP", code = "The Debater" },
        new PersonalityType { name = "ENTJ", code = "The Commander" }
    };

    // Example method to get a personality type by its code
    public PersonalityType GetPersonalityTypeByCode(string code)
    {
        foreach (PersonalityType type in personalityTypes)
        {
            if (type.code == code)
            {
                return type;
            }
        }
        return new PersonalityType(); // Return an empty type if not found
    }

    // Example method to get a personality type by its name
    public PersonalityType GetPersonalityTypeByName(string name)
    {
        foreach (PersonalityType type in personalityTypes)
        {
            if (type.name == name)
            {
                return type;
            }
        }
        return new PersonalityType(); // Return an empty type if not found
    }
}
