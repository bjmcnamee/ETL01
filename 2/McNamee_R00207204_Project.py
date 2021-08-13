"""
import os
os.getcwd()
os.chdir('/home/barnyard/0python/college/MTU_Python_Projects/2')
os.getcwd()
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
      print (f)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

demo = pd.read_csv("NhanesDemoAdapted.csv")
# demo.describe()
food = pd.read_csv("NhanesFoodAdapted.csv")
# food.describe()

def GetMenuOption():
      x = ""; menulist = [1,2,3,4,5]
      # display menu and accept input within allowed range
      while x not in menulist:
            print("\n\nPlease select one of the following options 1 - 5 :")
            print("1 Household income per ethnicity")
            print("2 Marital status")
            print("3 Income and education level")
            print("4 Diet analysis")
            print("5 Exit")
            x = int(input()) # convert console string input to integer
            if x not in menulist:
                  print("NB Only number 1, 2, 3, 4 or 5 accepted!\n") # error prompt
      return(x)

def GetIncomeByEthnicity(demo):
      print("\nHousehold Income per ethnicity")
      print("------------------------------")
      print("Ethnicity\t\t# Respondents")
      # count of values for each Ethnicity (and sorted by index)
      df = demo['Ethnicity'].value_counts().sort_index()
      print(df)
      print("------------------------------------------\n")
      # demo df grouped by Ethnicity --> mean of groups --> sort group means by index
      groupdemo = demo.groupby("Ethnicity")
      df = round(groupdemo['HouseholdIncome'].mean().sort_index(),2)
      print("Average Household Income")
      print("------------------------")
      print(df)
      print("------------------------------------------")
      # PLOT horizontal bar graph for Household Income v Ethnicity
      df.plot(kind='barh') # plot transformed dataframe with title + labels
      plt.title('Household Income per ethnicity',fontweight='bold')
      plt.xlabel('Average Household Income',fontweight='bold')
      plt.ylabel('Ethnicity',fontweight='bold')
      plt.show()
      Main()
      
def GetMaritalStatusByAge(demo):
      print("\n\nMarital status (age>20)")
      print("------------------------------------------")
      print("Status # Respondents")
      # demo df reduced to MS column --> subset by Age var > 20 --> count of each MS status --> sorted by index 
      df = demo['Marital Status'][demo['Age']>20].value_counts().sort_index()
      print("------------------------------------------")
      # demo df reduced to MS and Age columns --> subset by Age var > 20 --> sorted by index 
      df = demo[['Marital Status','Age']][demo['Age']>20].sort_index()
      # PLOT line frequency graph for Marital Status v Count
      plt.title('Marital status (age>20)',fontweight='bold')
      ax = sns.boxplot(x='Marital Status', y='Age', data = df)
      text = '1.0 : Married\n2.0 : Widowed\n3.0 : Divorced\n4.0 : Separate\n5.0 : Single\n6.0 : Living with Partner'
      props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
      ax.text(1.1, 0.95, text, transform=ax.transAxes, fontsize=14,
      verticalalignment='top', bbox=props)
      plt.show()
      Main()
      
def GetIncomeByEducation(demo):
      print("Income and education level\n")
      # display console menu with 2 options 1,2 and prompt for only correct values input
      x = ""; menulist = [1,2]
      while x not in menulist:
            print("Please select one of the following options 1 - 2 :")
            print("1 Income Poverty Ratio")
            print("2 Household Income")
            x = int(input())
            if x not in menulist:
                  print("NB Only number 1 or 2 accepted!\n")
      if x == 1:
            print("Income Poverty Ratio\n")
            # demo df grouped by Education into series --> add mean of IPR column --> round to .00 --> remove the first row of null values (empty csv cells)
            series = round(demo.groupby('Education')[['IncomePovertyRatio']].mean().tail(5),2)
            print(series)
            # PLOT simple bar graph for Income Poverty Ratio v Education level
            series.plot(kind='bar')
            plt.title('Income Poverty Ratio v Education',fontweight='bold')
            plt.xlabel('Education',fontweight='bold')
            plt.ylabel('Income Poverty Ratio',fontweight='bold')
            plt.text(6, 0, '1.0 : 9th Grade\n2.0 : 9-11th grade\n3.0 : HighSchool Grad\n4.0 : Some college\n5.0 : College Grad or above')
            plt.show()
            Main()
      elif x == 2:
            print("Household Income\n")
            # demo df grouped by Education into series --> add mean of HI column --> round to .00 --> remove the first row of null values (empty csv cells)
            series = round(demo.groupby('Education')[['HouseholdIncome']].mean().iloc[1:6],2)
            print(series)
            # PLOT simple bar graph for Household Income v Education level
            series.plot(kind='bar')
            plt.title('Household Income v Education',fontweight='bold')
            plt.xlabel('Education',fontweight='bold')
            plt.ylabel('Household Income',fontweight='bold')
            plt.text(6, 0, '1.0 : 9th Grade\n2.0 : 9-11th grade\n3.0 : HighSchool Grad\n4.0 : Some college\n5.0 : College Grad or above')
            plt.show()
            Main()
            
def GetDietAnalysis(demo,food):
      # groupby food data per individual --> average for the week's recordings in new dataframe 
      dfgroup = round(food.groupby('SEQN').mean(),2)
      dfreduce = dfgroup.drop(columns=['dGRMS', 'dKCAL'], axis=0)
      dfmerged = pd.merge(demo,dfreduce,on='SEQN')
      dfmerged.to_csv('McNamee_R00207204_Merged.csv')
      print("\nDemo and Food dataframes merged successfully to 'McNamee_R00207204_Merged.csv'\n")
      # generate new menu
      print("Select one of the following variables to see merged dataset views of Gender, Ethnicity and Education or Household Income and Age")
      x = ""; menulist = list(range(1,12))
      columnlist = food.columns.tolist()[3:-1]
      for i in range(0,len(columnlist)):
            print(i+1,columnlist[i])
      while x not in menulist :
            x = int(input())
            if x not in menulist :
                  print("any number between 1 and 11 only")
      SelectedFood = columnlist[x-1]
      
      # df merged --> HI, Age, SelectedFood
      df2 = dfmerged[['HouseholdIncome','Age',SelectedFood]]
      # generate scatterplot HI v Age
      # create 3rd var size to represent SelectedFood level
      SelectedFoodValues = df2.iloc[:,2].tolist()
      size = [20*SelectedFoodValues[n] for n in range(len(SelectedFoodValues))]
      df2.plot.scatter(x='Age', y='HouseholdIncome',s=size)
      title = 'Household Income v Age (' + SelectedFood + ')'
      plt.title(title,fontweight='bold')
      title = "Bubble size = " + SelectedFood + " range : " + str(min(SelectedFoodValues)) + " to " + str(max(SelectedFoodValues))
      plt.legend(title=title)
      plt.xlabel('Age',fontweight='bold')
      plt.ylabel('Household Income',fontweight='bold')
      plt.show()
      
      
      x = [0,2,4,6,8,10]
      y = [0]*len(x)
      size = [20*4**n for n in range(len(x))]
      plt.scatter(x,y,s=s)
      plt.show()
      
      # df merged --> Gender, Ethnicity, Education & SelectedFood
      df1 = dfmerged[['Gender','Ethnicity','Education',SelectedFood]]
      df1 = pd.DataFrame(data = dfmerged, columns = ['Gender','Ethnicity','Education',SelectedFood])
      
      # generate boxplot Gender v SelectedFood
      ax = sns.boxplot(x='Gender', y=SelectedFood, data=df1, showfliers = False)
      title = 'Gender v ' + SelectedFood
      plt.title(title,fontweight='bold')
      plt.xlabel('Gender',fontweight='bold')
      plt.ylabel(SelectedFood,fontweight='bold')
      plt.show()
      # generate boxplot Ethnicity v SelectedFood
      ax = sns.boxplot(x='Ethnicity', y=SelectedFood, data=df1, showfliers = False)
      ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
      title = 'Ethnicity v ' + SelectedFood
      plt.title(title,fontweight='bold')
      plt.xlabel('Ethnicity',fontweight='bold')
      plt.ylabel(SelectedFood,fontweight='bold')
      plt.show()
      # generate boxplot Education v SelectedFood
      ax = sns.boxplot(x='Education', y=SelectedFood, data=df1, showfliers = False)
      title = 'Education v ' + SelectedFood
      plt.title(title,fontweight='bold')
      plt.xlabel('Education',fontweight='bold')
      plt.ylabel(SelectedFood,fontweight='bold')
      plt.text(5, 0, '1.0 : 9th Grade\n2.0 : 9-11th grade\n3.0 : HighSchool Grad\n4.0 : Some college\n5.0 : College Grad or above')
      plt.show()
      
      
      Main()
      
def Main():      
      choice = GetMenuOption()
      if choice == 1:
            GetIncomeByEthnicity(demo)
      elif choice == 2:
            GetMaritalStatusByAge(demo)
      elif choice == 3:
            GetIncomeByEducation(demo)      
      elif choice == 4:
            GetDietAnalysis(demo,food)
      else :
            print("Goodbye")
      
     
Main()