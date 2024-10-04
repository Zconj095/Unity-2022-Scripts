def print_menu():
    print('1. Milimeters to Centimeters')
    print('2. Centimeters to Meters')
    print('3. Meters to Kilometers')
    print('4. Kilometers to Inches')
    print('5. Inches to Feet')
def mm_centimeters():
    mm = float(input('Enter a distance in milimeters: '))
    centimeters = mm * 0.1
    print('You Entered {0}. Distance in centimeters: {1}'.format(mm, centimeters))
    print('Distance in milimeters: {0}'.format(mm))
    
def cm_meters():
    cm = float(input('Enter a distance in centimeters: '))
    meters = cm * 0.01
    print('You Entered {0}. Distance in meters: {1}'.format(cm, meters))
    print('Distance in meters: {0}'.format(meters))
    
def m_kilometers():
    m = float(input('Enter a distance in meters: '))
    kilometers = m * 0.001
    print('You Entered {0}. Distance in kilometers: {1}'.format(m, kilometers))
    print('Distance in kilometers: {0}'.format(kilometers))
print_menu()

def km_inches():
    km = float(input('Enter a distance in kilometers: '))
    inches = km * 39370.07874
    print('You Entered {0}. Distance in inches: {1}'.format(km, inches))
    print('Distance in inches: {0}'.format(inches))

def inch_feet():
    inch = float(input('Enter a distance in inches: '))
    feet = inch * 0.08333
    print('You Entered {0}. Distance in feet: {1}'.format(inch, feet))
    print('Distance in feet: {0}'.format(feet))

choice = input('which conversion would you like to do?: ')

if choice == '1':
    mm_centimeters()
if choice == '2':
    cm_meters()
if choice == '3':
    m_kilometers()
if choice == '4':
    km_inches()
if choice == '5':
    inch_feet()
