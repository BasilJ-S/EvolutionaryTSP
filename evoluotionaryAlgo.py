

# Code to implement evolutionary algorithm.
import numpy as np
import pandas as pd
from typing import NewType

class Individual:
    def __init__(self, path: np.array):
        self.path = np.array(path)
        
    def produceOffspring(self, other: 'Individual') -> 'Individual':
        # Assuming single-point crossover for simplicity
        crossover_points = np.random.randint(1, len(self.path) - 1, size = (1,2))
        crossover_point1 = np.min(crossover_points)
        crossover_point2 = np.max(crossover_points)
        if crossover_point1 != crossover_point2:

            # We will cross over the interval of the other into the interval of the self
            intervalReplaced = self.path[crossover_point1:crossover_point2]
            intervalReplacing = other.path[crossover_point1:crossover_point2]

            # Convert lists to numpy arrays
            intervalReplaced = np.array(intervalReplaced)
            intervalReplacing = np.array(intervalReplacing)
            
            # logical array, true if an element of first argument is NOT in the second argument
            #missingInNewIndividual = np.isin(intervalReplaced, intervalReplacing, invert=True)
            missingInNewIndividual = np.invert((intervalReplaced[:, None] == intervalReplacing).all(-1).any(-1))

            # If it is in the replaced array but not the replacing array, we will lose a city and not visit all cities
            # i.e. we need to add these cities to the new individual (outside of the range)
            toBeAdded = intervalReplaced[missingInNewIndividual]

            extraInNewIndividual = np.invert((intervalReplacing[:, None] == intervalReplaced).all(-1).any(-1))
            # If it is in the replacing array, but not the replaced array, we have duplicates of that element
            # i.e. we need to remove these cities from the new individual (outside of the range)
            toBeReplaced = intervalReplacing[extraInNewIndividual]

            if len(toBeAdded) != len(toBeReplaced):
                print(crossover_points)
                raise Exception("Length of duplicates in new individual not equal to length of missing values in new individual.")
            
            # Create new individual path with potential errors
            newIndividualPath = np.concatenate((self.path[:crossover_point1], other.path[crossover_point1:crossover_point2], self.path[crossover_point2:]))
            #print(newIndividualPath)
            # Find locations where we can put in missing values
            maskOfReplacePoints = ((self.path[:, None] == toBeReplaced).all(-1).any(-1))
            #print(maskOfReplacePoints)
            # Inset missing values
            newIndividualPath[maskOfReplacePoints] = toBeAdded
            #print(newIndividualPath)

            if np.random.rand() > 0.995:
                middle: int = int(np.floor(len(newIndividualPath)/2))
                newIndividualPath = np.concatenate((newIndividualPath[middle:], newIndividualPath[:middle]))
            
            return Individual(newIndividualPath)  
        else:
            return self
    
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

    
    def getScore(self,path: np.array) -> float:
        if len(path) != len(self.cities):
            raise Exception(f"Path length ({len(path)}) is not the same length as the total number of cities ({len(self.cities)})")
        else:
            # Should theoretically check that the path is a valid path, but for efficiency this is not 
            #print(path)
            distance: list[float] = np.linalg.norm(np.subtract(path[1:][:],path[:len(path)-1][:]), axis=1)
            total = np.sum(distance)

            return total
        
    def getScores(self,individuals: list[Individual]) -> float:
        scores: list[float] = []
        for ind in individuals:
            scores.append(self.getScore(ind.path))

        return scores
            
    def getCityOrder(self,path: np.array) -> list[int]:
        if len(path) != len(self.cities):
            raise Exception(f"Path length ({len(path)}) is not the same length as the total number of cities ({len(self.cities)})")
        cityPath:list[int] = []
        for i in range(len(self.cities)):
            try:
                cityPath.append(self.locations[f"{path[i][0]},{path[i][1]}"])
            except:
                raise Exception(f"City at ({path[i][0]},{path[i][1]}) does not exist.")
        return cityPath
    
    def generateRandomIndividual(self, randomNumberGenerator: np.random) -> Individual:
        shuffled = np.copy(self.cities)
        randomNumberGenerator.shuffle(shuffled, axis = 0)
        #print(shuffled)
        return Individual(shuffled)
    
    def generateRandomPopulation(self,populationSize: int) -> list[Individual]:
        randomNumberGenerator = np.random.default_rng()
        individuals: list[Individual] = []
        for i in range(populationSize):
            individuals.append(self.generateRandomIndividual(randomNumberGenerator))

        return individuals

class Population:
    def __init__(self, individuals: list[Individual]):
        self.individuals = individuals
        self.populationSize = len(self.individuals)
    
    # Return new population, along with the best score and best individual from the current population
    def getNewPopulation(self, path: Path) -> tuple['Population', float, Individual]:
        fitness = path.getScores(self.individuals)
        bestIndividual = np.argmin(fitness)
        worstScore = np.max(fitness)
        bestScore = fitness[bestIndividual]

        fitness = np.abs(np.divide(np.subtract(fitness,worstScore),worstScore - bestScore))
        print(fitness)
        print("-----")

        scoreSum = np.sum(fitness)

        probOfReproducing: list[float] = np.divide(fitness,scoreSum)
        population: list[int] = range(self.populationSize)
        
        randomNumberGenerator = np.random.default_rng()
        newPopulation: list[Individual] = []

        for i in range(self.populationSize):
            mates = randomNumberGenerator.choice(population,(1,2),True,probOfReproducing)
            newIndividual = self.individuals[mates[0][0]].produceOffspring(self.individuals[mates[0][1]])
            newPopulation.append(newIndividual)

        return Population(newPopulation), bestScore, self.individuals[bestIndividual]
        



if __name__ == "__main__":
    x = Path([1,2,3],[1,2,3])
    possiblePath = [[1,1],[3,3],[2,2]]
    print(x.getCity(3,3))
    print(x.getLocation(2))
    print(x.getScore(possiblePath))
    print(x.getCityOrder(possiblePath))

    # READ IN FILE
    file = pd.read_csv('./ch130.tsp', skiprows=6, names = ['city','x','y'], delimiter=' ', skipfooter=1)
    file = file
    print(file.head(10))
    x_coords = file['x'].tolist()
    y_coords = file['y'].tolist()
    path = Path(x_coords, y_coords) 


    ind = Individual([[1,1],[3,3],[2,2],[4,4],[5,5],[6,6]])
    ind2 = Individual([[1,1],[4,4],[2,2], [6,6],[5,5],[3,3]])

    ind3 = ind.produceOffspring(ind2)

    print("PROPER TRIALS")

    population = Population(path.generateRandomPopulation(10))
    bestIndividual: Individual = None
    bestScores = []
    for i in range(10000):
        try:
            population, bestScore, bestIndividual = population.getNewPopulation(path)
            bestScores.append(bestScore)
        except:
            print("NO DIVERSITY")
            break
    
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(bestScores)
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Best Score')
    axs[0].set_title('Best Score by Iteration')

    axs[1].scatter(path.cities[:,0], path.cities[:,1])
    axs[1].plot(bestIndividual.path[:, 0], bestIndividual.path[:, 1])
    axs[1].set_title('Best Path')

    plt.tight_layout()
    plt.show()



    




        
