import pandas as pd
df = pd.read_excel (r'C:\Users\zconj\OneDrive\Documents\New folder\New Unity Project\Assets\TESTDATAFORPYTHONSCRIPT.xlsx', sheet_name='RESEARCH')
a = df['FORMS_OF_EMPHASIS']
b = df['TYPES_OF_VOCALS']
c = df['TYPES_OF_ENERGY']
d = df['CHEMISTRY']
e = df['CLASSES_OF_VOCALS']
f = df['CLASSES_OF_ENERGY']
g = df['CONNECTION_TO_TYPES_OF_VOCALS_AND_CHAKRAS']
h = df['CLASSES_OF_BREATHING_PATTERNS']
print("A NEW FORM OF EMPHASIS KNOWN AS",df.FORMS_OF_EMPHASIS[1],"HAS BEEN DISCOVERED")
print("******************************************************************")








