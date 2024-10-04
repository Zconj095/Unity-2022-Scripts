import PyPDF2 as PyPDF2
from PyPDF2 import PdfFileReader

#Creating a PDF file object
pdf = open("C:\PDF_Files\Unlocking_the_Brain_Coding.pdf","rb")

#creating pdf reader object
pdf_reader = PyPDF2.PdfFileReader(pdf)

#checking number of pages in the pdf file
print(pdf_reader.numPages)

#Creating a page object
page = pdf_reader.getPage(0)

#extracting text from the page
print(page.extractText())

#close the pdf file
pdf.close()
