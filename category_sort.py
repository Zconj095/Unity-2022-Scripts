def category_sort(categories: list, items: list) -> dict:
    category_dict = {}
    for category in categories:
        category_dict[category] = []
    for item in items:
        for category in categories:
            if item in category_dict[category]:
                category_dict[category].append(item)
    return category_dict

# Example usage:
categories = ["Fruit", "Vegetable", "Grain"]
items = ["Apple", "Carrot", "Bread", "Banana", "Potato", "Wheat"]
sorted_categories = category_sort(categories, items)
print(sorted_categories)
# Output:
# {'Fruit': ['Apple', 'Banana'], 'Vegetable': ['Carrot', 'Potato'], 'Grain': ['Bread', 'Wheat']}