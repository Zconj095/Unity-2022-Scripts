def seasons():
    return {
        'Spring': {'start': 'March 20', 'end': 'June 20'},
        'Summer': {'start': 'June 21', 'end': 'September 22'},
        'Autumn': {'start': 'September 23', 'end': 'December 21'},
        'Winter': {'start': 'December 22', 'end': 'March 19'}
    }

# Example usage:
seasons_dict = seasons()
print(seasons_dict['Summer']['start'])  # Output: June 21
print(seasons_dict['Winter']['end'])  # Output: March 19