##CODE: Julian C.
import math as md
import random as rd
##Runs input as if it were each line of image was cut into n chunks, where n is SEGMENTS
LEARNING_RATE=0.1
RANGE_OF_RANDOM=0.1
TRAINING_EPOCHS=5
SEGMENTS=8
LEN=int(32/SEGMENTS)

class Image():
    def __init__(self,rd_file,bounds):
        #rd_file for data set
        #bounds in the form of [x1,x2,pos],
        #where x1 and x2 are the start and end of the image,
        #and pos is the position of the label
        ##builds image with data for printing, analyzing, and labeling
        self.print_data=rd_file[bounds[0]:bounds[1]+1]
        string=""
        for pos in range(bounds[0],bounds[1]+1):   
            string+=rd_file[pos]
        self.data=string
        self.label=rd_file[bounds[2]]

    def get_digit(self,pos):
        #returns digit from a position in self.data
        return int(self.data[pos])

    def get_average(self,pos,n):
        #returns the average of digits from a slice of n length in self.data
        sum_digs=0
        for x in range(pos,pos+n):
            sum_digs+=int(self.data[x])
        return float(sum_digs/n)


    def get_label(self):
        ##returns a label for the image
        return int(self.label)

class Node():
    def __init__(self,len_input_layer):
        ##initializes weights and value of node
        self.value=None
        self.weights=[]
        for i in range(len_input_layer):
            self.weights.append(rd.uniform(-RANGE_OF_RANDOM,RANGE_OF_RANDOM))
       
    def run(self,image_data,pos=None,list_nodes=[]):
        if not pos is None and len(list_nodes)==0:
            ##sets value to a data point in image
            ## ie input node setup
            self.value=image_data.get_average(pos,LEN)
        elif len(list_nodes)>0 and pos is None:
            ##calculates the node's value when given an input node list
            ## ie output node setup
            sum=0 
            for i in range(len(list_nodes)): 
                sum+=list_nodes[i].value*self.weights[i]
            self.value=sigmoid(sum)
        else:
            ##bias node setup
            self.value=1
    
    def learn(self,list_nodes,error):
        ##modifies weights based on several factors, ie learning
        learning_rate=LEARNING_RATE
        for i in range(len(self.weights)): 
            modifier=learning_rate*error*self.value*(1-self.value)*list_nodes[i].value
            self.weights[i]=self.weights[i]+modifier
    
def sigmoid(num):
    return 1/(1+md.exp(-1*num))
      
def read_in(filename):
    ##reads in file and outputs in a form of a list
    output=[]
    with open(filename, "r") as file:
        for line in file:
            form_line=line.strip().strip("/n")
            output.append(form_line)
    return output

def print_file(list_data):
    ##prints out print data in readible form
    for string in list_data:
        print(string)

def run_alg (input_layer,output_layer,image):
    ## runs input layer setup and weight calculation before comparing with image label
        for num in range(0,len(image.data),LEN):
            input_layer[int(num/LEN)].run(image,pos=num)
        input_layer[len(input_layer)-1].run(image)
        highest=0
        pos=0
        ##finds the output of the algorithm and compares it with the label of the image
        for i in range(len(output_layer)):
            output_layer[i].run(image,list_nodes=input_layer)
            if output_layer[i].value>=highest:
                highest=output_layer[i].value
                pos=i
        if pos==image.get_label():
            return 1
        return 0

def back_propagate(input_layer,output_layer,image):
    ##learning segment
    label_data=[0]*len(output_layer)
    label_data[image.get_label()]=1
    for i in range(len(output_layer)):
        error=label_data[i]-output_layer[i].value
        output_layer[i].learn(input_layer,error)

def main():
    rd_file=read_in("optdigits-32x32.tra")
    len_image=33
    images=[]
    ##inserts all data into images for training data
    for pos in range(3,len(rd_file),len_image):
        imr= Image(rd_file,[pos,pos+len_image-2,pos+len_image-1])
        images.append(imr)
    ##makes all the input nodes
    input_layer=[]
    for num in range(int(len(images[0].data)/(LEN))):
        input_node= Node(0)
        input_layer.append(input_node)
    input_node= Node(0)
    input_layer.append(input_node)
    ##makes all the output nodes
    output_layer=[]
    for num in range(10):
        output_node= Node(len(input_layer))
        output_layer.append(output_node)
    ## the training and learning segment for training data
    print("Training Data:")
    for epoch in range(TRAINING_EPOCHS):
        print("Epoch "+str(epoch+1)+":")
        num_correct=0
        num_tot=0
        rd.shuffle(images)
        for image in images:
            num_correct+=run_alg(input_layer,output_layer,image)
            num_tot+=1
            back_propagate(input_layer,output_layer,image)
        print("Accuracy: "+str(num_correct/num_tot*100) + "%")
    ##inserts all data into images for testing data
    rd_test=read_in("optdigits-32x32.tes")
    test_images=[]
    for pos in range(3,len(rd_file),len_image):
        imr= Image(rd_file,[pos,pos+len_image-2,pos+len_image-1])
        test_images.append(imr)
    print("Testing Data:")
    ## the testing segment for testing data
    num_correct=0
    num_tot=0
    for image in test_images:
        num_correct+=run_alg(input_layer,output_layer,image)
        num_tot+=1
    print("Accuracy: "+str(num_correct/num_tot*100) + "%")
main()
