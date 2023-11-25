
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from IPython import display as ipythondisplay
from IPython.display import clear_output
import time

class Cube():
    moveDict = {'Clockwise Right' : 'R',
                'Counter-Clockwise Right' : 'R\'',
                'Clockwise Left' : 'L',
                'Counter-Clockwise Left' : 'L\'',
                'Clockwise Front' : 'F',
                'Counter-Clockwise Front' : 'F\'',
                'Clockwise Back' : 'B',
                'Counter-Clockwise Back' : 'B\'',
                'Clockwise Up' : 'U',
                'Counter-Clockwise Up' : 'U\'',
                'Clockwise Down' : 'D',
                'Counter-Clockwise D' : 'D\''}
    
    
    colDict = {'w': '#DBDBDB',
           'y': '#DBD923',
           'r': '#DB2115',
           'o': '#DB5616',
           'g': '#40DB51',
           'b': '#485DDB'}
    
    faceDict = {'F' : 'front',
                'B' : 'back',
                'U' : 'up',
                'D' : 'down',
                'L' : 'left',
                'R' : 'right'}
    
    #Binary Representation for State/Observation Representation in RL
    ColorCode = { 'w' : 0,
                    'y' : 1,
                    'r' : 2,
                    'o' : 3,
                    'g' : 4,
                    'b' : 5}
    
    def __init__(self):
        #Initialization of all faces
        self.U = np.array([['g' for i in range(3)] for j in range(3)])
        self.D = np.array([['b' for i in range(3)] for j in range(3)])
        self.F = np.array([['w' for i in range(3)] for j in range(3)])
        self.B = np.array([['y' for i in range(3)] for j in range(3)])
        self.R = np.array([['o' for i in range(3)] for j in range(3)])
        self.L = np.array([['r' for i in range(3)] for j in range(3)])

    def plot_face(self,ax, matrix, color_dict):
        for i in range(3):
            for j in range(3):
                color = color_dict.get(matrix[i][j], 'black')
                rect = plt.Rectangle((j, 2 - i), 1, 1, linewidth=2, edgecolor='black', facecolor=color)
                ax.add_patch(rect)

        # Add grid lines
        for i in range(1, 3):
            ax.axhline(i, color='black', linewidth=2)
            ax.axvline(i, color='black', linewidth=2)

        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_xticks([])
        ax.set_yticks([])
       
             
    def render(self, isColor=True):
        fig, axs = plt.subplots(4, 4, figsize=(6, 6))

        self.plot_face(axs[0, 1],self.U, Cube.colDict)
        self.plot_face(axs[1, 0],self.L, Cube.colDict)
        self.plot_face(axs[1, 1],self.F, Cube.colDict)
        self.plot_face(axs[1, 2],self.R, Cube.colDict)
        self.plot_face(axs[1, 3],self.B, Cube.colDict)
        self.plot_face(axs[2, 1],self.D, Cube.colDict)
        # Remove the empty subplot
        axs[0, 0].axis('off')
        axs[0, 2].axis('off')
        axs[0, 3].axis('off')
        axs[2, 0].axis('off')
        axs[2, 3].axis('off')
        axs[2, 2].axis('off')
        axs[3, 0].axis('off')
        axs[3, 1].axis('off')
        axs[3, 2].axis('off')
        axs[3, 3].axis('off')

        #plot for 2 seconds then delete the plot
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        
        
         
    
    def do(self, move):
        
        up = np.array(self.U.copy())
        down = np.array(self.D.copy())
        left = np.array(self.L.copy())
        right = np.array(self.R.copy())
        front = np.array(self.F.copy())
        back = np.array(self.B.copy())
        
        if move == 'U':
            #Rotation of Up
            N = len(up[0])
            for i in range(N // 2):
                for j in range(i, N - i - 1):
                    temp = up[i][j]
                    up[i][j] = up[N - 1 - j][i]
                    up[N - 1 - j][i] = up[N - 1 - i][N - 1 - j]
                    up[N - 1 - i][N - 1 - j] = up[j][N - 1 - i]
                    up[j][N - 1 - i] = temp
            self.U = up
            
            #Rotation of Rest:
            self.L[0] = front[0]
            self.B[0] = left[0]
            self.R[0] = back[0]
            self.F[0] = right[0]
            
        elif move == 'U\'':
            #Rotation of Up
            self.U = np.array([[up[j][i] for j in range(len(up))] for i in range(len(up[0])-1,-1,-1)])
            
            #Rotation of Rest
            self.F[0] = left[0]
            self.L[0] = back[0]
            self.B[0] = right[0]
            self.R[0] = front[0]
        
        elif move == 'D':
            #Rotation of Down
            N = len(down[0])
            for i in range(N // 2):
                for j in range(i, N - i - 1):
                    temp = down[i][j]
                    down[i][j] = down[N - 1 - j][i]
                    down[N - 1 - j][i] = down[N - 1 - i][N - 1 - j]
                    down[N - 1 - i][N - 1 - j] = down[j][N - 1 - i]
                    down[j][N - 1 - i] = temp
            self.D = down
            
            #Rotation of Rest:
            self.L[2] = back[2]
            self.B[2] = right[2]
            self.R[2] = front[2]
            self.F[2] = left[2]
        
        elif move == 'D\'':
            #Rotation of Down
            self.D = np.array([[down[j][i] for j in range(len(down))] for i in range(len(down[0])-1,-1,-1)])
            
            #Rotation of Rest
            self.L[2] = front[2]
            self.B[2] = left[2]
            self.R[2] = back[2]
            self.F[2] = right[2]
        
        elif move == 'R':
            #Rotation of Right
            N = len(right[0])
            for i in range(N // 2):
                for j in range(i, N - i - 1):
                    temp = right[i][j]
                    right[i][j] = right[N - 1 - j][i]
                    right[N - 1 - j][i] = right[N - 1 - i][N - 1 - j]
                    right[N - 1 - i][N - 1 - j] = right[j][N - 1 - i]
                    right[j][N - 1 - i] = temp
            self.R = right
            
            #Rotation of Rest:
            self.U[:,2] = front[:,2]
            self.B[:,0] = up[::-1,2]
            self.D[:,2] = back[::-1,0]
            self.F[:,2] = down[:,2]
            
        elif move == 'R\'':
            #Rotation of Right
            self.R = np.array([[right[j][i] for j in range(len(right))] for i in range(len(right[0])-1,-1,-1)])
            
            #Rotation of Rest
            self.U[:,2] = back[::-1,0]
            self.B[:,0] = down[::-1,2]
            self.D[:,2] = front[:,2]
            self.F[:,2] = up[:,2]
            
        elif move == 'L':
            #Rotation of Left
            N = len(left[0])
            for i in range(N // 2):
                for j in range(i, N - i - 1):
                    temp = left[i][j]
                    left[i][j] = left[N - 1 - j][i]
                    left[N - 1 - j][i] = left[N - 1 - i][N - 1 - j]
                    left[N - 1 - i][N - 1 - j] = left[j][N - 1 - i]
                    left[j][N - 1 - i] = temp
            self.L = left
            
            #Rotation of Rest:
            self.U[:,0] = back[::-1,2]
            self.B[:,2] = down[::-1,0]
            self.D[:,0] = front[:,0]
            self.F[:,0] = up[:,0]
            
        elif move == 'L\'':
            #Rotation of Left
            self.L = np.array([[left[j][i] for j in range(len(left))] for i in range(len(left[0])-1,-1,-1)])
            
            #Rotation of Rest
            self.U[:,0] = front[:,0]
            self.B[:,2] = up[::-1,0]
            self.D[:,0] = back[::-1,2]
            self.F[:,0] = down[:,0]
        
        elif move == 'F':
            #Rotation of front
            N = len(front[0])
            for i in range(N // 2):
                for j in range(i, N - i - 1):
                    temp = front[i][j]
                    front[i][j] = front[N - 1 - j][i]
                    front[N - 1 - j][i] = front[N - 1 - i][N - 1 - j]
                    front[N - 1 - i][N - 1 - j] = front[j][N - 1 - i]
                    front[j][N - 1 - i] = temp
            self.F = front
            
            #Rotation of Rest:
            self.U[2] = left[::-1,2]
            self.R[:,0] = up[2]
            self.D[0] = right[::-1,0]
            self.L[:,2] = down[0]
            
        elif move == 'F\'':
            #Rotation of Front
            self.F = np.array([[front[j][i] for j in range(len(front))] for i in range(len(front[0])-1,-1,-1)])
            
            #Rotation of Rest
            self.U[2] = right[:,0]
            self.R[:,0] = down[0,::-1]
            self.D[0] = left[:,2]
            self.L[:,2] = up[2,::-1]
        
        elif move == 'B':
            #Rotation of Back
            N = len(back[0])
            for i in range(N // 2):
                for j in range(i, N - i - 1):
                    temp = back[i][j]
                    back[i][j] = back[N - 1 - j][i]
                    back[N - 1 - j][i] = back[N - 1 - i][N - 1 - j]
                    back[N - 1 - i][N - 1 - j] = back[j][N - 1 - i]
                    back[j][N - 1 - i] = temp
            self.B = back
            
            #Rotation of Rest:
            self.U[0,:] = right[:,2]
            self.R[:,2] = down[2,::-1]
            self.D[2,:] = left[:,0]
            self.L[:,0] = up[0,::-1]
            
        elif move == 'B\'':
            #Rotation of Back
            self.B = np.array([[back[j][i] for j in range(len(back))] for i in range(len(back[0])-1,-1,-1)])
            
            #Rotation of Rest
            self.U[0,:] = left[::-1,0]
            self.R[:,2] = up[0,:]
            self.D[2,:] = right[::-1,2]
            self.L[:,0] = down[2,:]
    
    def shuffle(self, number_of_moves = 1, seed = 7):
        for i in range(number_of_moves):
            move = random.choice(list(self.moveDict.values()))
            self.do(move)
    
    def reset(self):
        self.__init__()
    
    def get_array(self):
        vector = []
        faces = [ self.U, self.L,self.F, self.R,self.B, self.D]
        for face in faces:
            for faceRow in face:
                for faceTile in faceRow:              
                    vector.append(self.ColorCode[faceTile])
        return vector
    
    def isdone(self):
        myarray = self.get_array()
        correct_array = Cube().get_array()
        for i in range(len(myarray)):
            if myarray[i]!=correct_array[i]:
                return False
        return True
    
    def solvability_percentage(self):
   
        myarray = self.get_array()
        correct_array = Cube().get_array()
        correct_stickers = sum(1 for current, solved in zip(myarray, correct_array) if current == solved)

        # Calculate the solvability percentage
        total_stickers = len(myarray)
        solvability_percentage = (correct_stickers / total_stickers) * 100

        return solvability_percentage

    
 
