

# Code to implement evolutionary algorithm.
import numpy as np
import pandas as pd

class Individual:
    def __init__(self, path, performance):
        self.path = path

    
    #def __calcPerformance(path):
        #for i,(x,y) 

# Class to hold information about a path and calculate useful results from a given path
class Path:
    def __init__(self, x: list[float], y:list[int]):
        if len(x) != len(y):
            raise Exception("Length of x not equal to length of y")
        
        self.length = len(x)
        self.cities = np.zeros((self.length,2))
        self.locations = {}
        for i in range(self.length):
            self.cities[i] = [x[i], y[i]]
            self.locations[f"{x[i]},{y[i]}"] = i
        

    def getLocation(self,index: int) -> tuple[float,float]:
        try:
            return self.cities[index]
        except:
            raise Exception(f"City {index} does not exist.")
    
    def getCity(self,x,y) -> int:
        try:
            return self.locations[f"{x},{y}"]
        except:
            raise Exception(f"No city exists at ({x},{y}).")

    
    def getScore(self,path: list[list[int]]) -> float:
        if len(path) != len(self.cities):
            raise Exception(f"Path length ({len(path)}) is not the same length as the total number of cities ({len(self.cities)})")
        else:
            # Should theoretically check that the path is a valid path, but for efficiency this is not 
            print(path)
            distance = np.linalg.norm(np.subtract(path[1:][:],path[:len(path)-1][:]), axis=1)
            total = np.sum(distance)

            return total
    
    def getCityOrder(self,path: list[list[int]]):
        if len(path) != len(self.cities):
            raise Exception(f"Path length ({len(path)}) is not the same length as the total number of cities ({len(self.cities)})")
        cityPath:list[int] = []
        for i in range(len(self.cities)):
            try:
                cityPath.append(self.locations[f"{path[i][0]},{path[i][1]}"])
            except:
                raise Exception(f"City at ({path[i][0]},{path[i][1]}) does not exist.")
        return cityPath


if __name__ == "__main__":
    x = Path([1,2,3],[1,2,3])
    possiblePath = [[1,1],[3,3],[2,2]]
    print(x.getCity(3,3))
    print(x.getLocation(2))
    print(x.getScore(possiblePath))
    print(x.getCityOrder(possiblePath))

    # READ IN FILE
    file = pd.read_csv('./ch130.tsp', skiprows=6, names = ['city','x','y'], delimiter=' ')
    file = file
    print(file.head(10))
    x_coords = file['x'].tolist()
    y_coords = file['y'].tolist()
    path = Path(x_coords, y_coords) 





        
