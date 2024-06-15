# List = [], ordered, duplicates ok, changable.
# List is close to a general array notion
# but it requires methods to be modified
exmapleOfList = [1, 2, 3, 6, 6, 7]
exmapleOfList.append(4)
exmapleOfList.remove(2)
print( "List: ", exmapleOfList, "\n\tCan find elem at index 0:", exmapleOfList[0], "\n\tLength:", len(exmapleOfList), "\n\tReturn index of elem 7:", exmapleOfList.index(7), "\n\tHow many 6 in the list:", exmapleOfList.count(6) )

# Tuple = (), ordered, duplicates ok, fully immutable.
# Tuple is a list that can't be changed.
# Utilize when you need a collection of elements with duplicates
# but you won't change it. Tuple is very fast btw
exampleOfTuple = (1, 2, 3, 6, 6, 4)
print( "\nTuple: ", exampleOfTuple, "\n\tCan find elem at index 0:" ,exampleOfTuple[0], "\n\tLength:", len(exampleOfTuple), "\n\tReturn index of elem 4:", exampleOfTuple.index(4), "\n\tHow many 6 in the tuple:", exampleOfTuple.count(6) )


# Set = {}, no order, no duplicates, elements can't be modified.
# Set is an unordered collection where each element in unique
# Can add and delete elements.
# Utilize when you need to check whether some value exists or not.
processes = {"process1","process6", "process5"}
processes.add("process1") # ignored since it is a duplicate
processes.remove("process5")
processes.add("process3")
print( "\nSet: ", processes, "\n\tCan NOT find elem by any index", "\n\tLength:", len(processes), "\n\tCheck if 'process1' exists in the set:", "process1" in processes )

# Dictionary = {"key": value}, ordered, no duplicates, changable.
# Set is an unordered collection where each element in unique
# Can add and delete elements.
# Utilize when you need to check whether some value exists or not.
capitals = {"China": "Beijin",
            "Russia": "Moscow",
            "USA": "Washington"}
capitals.update({"Germany": "Berlin"}) # add
capitals.update({"USA": "Washington D.C."}) # edit
capitals.pop("China") # delete

keys = capitals.keys() # get all keys
values = capitals.values() # get all values
pairs = capitals.items() # get each item

print( "\nDictionary:", capitals, "\n\tGet elem by the key 'USA': ", capitals.get("USA"), "\n\tLength:", len(capitals), "\n\tCheck if 'Japan' exists in the dict:", capitals.get("Japan") )

# To iterate over a dictionary:
print("\n\t- Iteration -")
for k, v in capitals.items():
  capitals.update({k: 'test'})
  print("\tFormer value of", k, ":", v, "\n\tNew value: ", capitals.get(k))

print("\n\tDictionary after iteration: ", capitals)

# ! Use help() to get methods and descriptions of each type of collection
# help(capitals)