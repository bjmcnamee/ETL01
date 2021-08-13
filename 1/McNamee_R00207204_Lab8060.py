import pandas as pd
import os
os.chdir('/home/barnyard/0python/college/MTU_Python_Projects/1')
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
      print (f)

def Q1ParseFile():
      filename = input("Pls enter name of file : ")
      # read all lines of file to list 'content'
      with open(filename) as f:
            content = f.readlines()
      dict={}

      # cycle through each line/element of list 'content'
      for line in range(0,len(content)):
            # remove eol char and whitespace (last two chars)
            cleanline = content[line].rstrip('\n')[:-2]
            # print("clean :",cleanline) - testing
            listline = cleanline.split(',')
            # print("list :",listline) - testing
            # if > 1 element in list then add to dict as key/value = element2/element1
            if len(listline) > 1:
                  # print("listline[0]",listline[0]) - testing
                  # print("listline[1]",listline[1]) - testing
                  dict[listline[1]]=listline[0]
                  # print("d :",dict) - testing
      # note, >2 elements was not explicitly mentioned so not added to dict but could be done easily with a loop to cycle through each element of list
      # note, lines with 1 element have been ignored as they were not explicitly requested
      return(dict)
      
def AutoByPandas():
      df = pd.read_csv("importsAuto.csv")    
      autotail(df)
      autopricelimits(df)
      autopricemax(df)
      
def autotail(df):      
      # prints out the last 10 rows for three columns of data: the make, engine-size and price of each car in the dataset
      dftail = df[['make','engine-size','price']].tail(10)
      print(dftail)
      
def autopricelimits(df):      
      # ask the user to specify an upper and lower price value, print out the make, engine-size, horsepower and price of all cars that meet that specification
      lowerlimit = int(input("Enter lower price limit : "))
      upperlimit = int(input("Enter upper price limit : "))
      dflower = df['price']>=lowerlimit
      dfupper = df['price']<=upperlimit
      dflimits = df[['make','engine-size','horsepower','price']][dflower & dfupper]
      print(dflimits)
      
def autopricemax(df):
      # calculate maximum selling price for each car make : print a line for each make and maximum price over all cars of that make, sorted in descending order by maximum price.
      dfmax = df.groupby('make')[['price']].max()
      print(dfmax)

def Main():
      Q1ParseFile()
      country = input("Please enter a location : ")
      if country in dict.keys() :
            print("The country associated with",country,"is",dict[country])
      else:
            print(country,"not found in dictionary")
      AutoByPandas()      
Main()
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
