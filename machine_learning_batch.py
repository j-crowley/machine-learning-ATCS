##CODE: Julian C.
import math as md
import random as rd

LEARNING_RATE=2
RANGE_OF_RANDOM=5
TRAINING_EPOCHS=10
BATCHES=100

class Image():
    def __init__(self,rd_file,bounds):
        #rd_file for data set
        #bounds in the form of [x1,x2,pos],
        #where x1 and x2 are the start and end of the image,
        #and pos is the position of the label
        self.print_data=rd_file[bounds[0]:bounds[1]+1]
        string=""
        for pos in range(bounds[0],bounds[1]+1):
            string+=rd_file[pos]
        self.data=string
        self.label=rd_file[bounds[2]]

    def get_digit(self,pos):
        return int(self.data[pos])

    def get_label(self):
        return int(self.label)

class Node():
    def __init__(self,len_input_layer):
        self.value=None
        self.weights=[]
        for i in range(len_input_layer):
            self.weights.append(rd.uniform(-RANGE_OF_RANDOM,RANGE_OF_RANDOM))
        ##needs weights for each output node
       
    def run(self,image_data,pos=None,list_nodes=[]):
        if not pos is None and len(list_nodes)==0:
            self.value=image_data.get_digit(pos)
        elif len(list_nodes)>0 and pos is None:
            sum=0 
            for i in range(len(list_nodes)): 
                sum+=list_nodes[i].value*self.weights[i]
            self.value=sigmoid(sum)
        else:
            self.value=1
    
    def learn(self,list_nodes,error):
        learning_rate=LEARNING_RATE
        for i in range(len(self.weights)): 
            modifier=learning_rate*error*self.value*(1-self.value)*list_nodes[i].value
            self.weights[i]=self.weights[i]+modifier
    
def sigmoid(num):
    return 1/(1+md.exp(-1*num))
      
def read_in(filename):
    output=[]
    with open(filename, "r") as file:
        for line in file:
            form_line=line.strip().strip("/n")
            output.append(form_line)
    return output

def print_file(list_data):
    for string in list_data:
        print(string)

def run_alg (input_layer,output_layer,image):
        for num in range(len(image.data)):
            input_layer[num].run(image,pos=num)
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

def batch_prop(input_layer,output_layer,images):
    cum_error=[0]*len(output_layer)
    for image in images:
        label_data=[0]*len(output_layer)
        label_data[image.get_label()]=1
        for i in range(len(output_layer)):
            cum_error[i]+=label_data[i]-output_layer[i].value
    for i in range(len(cum_error)):
        cum_error[i]=cum_error[i]/len(images)
        output_layer[i].learn(input_layer,cum_error[i])

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
    for num in range(len(images[0].data)+1):
        input_node= Node(0)
        input_layer.append(input_node)
    ##makes all the output nodes
    output_layer=[]
    for num in range(10):
        output_node= Node(len(images[0].data)+1)
        output_layer.append(output_node)
    print("Training Data:")
    for epoch in range(TRAINING_EPOCHS):
        print("Epoch "+str(epoch+1)+":")
        num_correct=0
        num_tot=0
        rd.shuffle(images)
        for x in range(BATCHES-1):
            for pos in range(int(x*len(images)/BATCHES),int((x+1)*len(images)/BATCHES)):
                num_correct+=run_alg(input_layer,output_layer,images[x])
                num_tot+=1
            batch_prop(input_layer,output_layer,images[int(x*len(images)/BATCHES):int((x+1)*len(images)/BATCHES)])
        print("Accuracy: "+str(num_correct/num_tot*100) + "%")
    ##inserts all data into images for testing data
    rd_test=read_in("optdigits-32x32.tes")
    test_images=[]
    for pos in range(3,len(rd_file),len_image):
        imr= Image(rd_file,[pos,pos+len_image-2,pos+len_image-1])
        test_images.append(imr)
    print("Testing Data:")
    num_correct=0
    num_tot=0
    for image in test_images:
        num_correct+=run_alg(input_layer,output_layer,image)
        num_tot+=1
    print("Accuracy: "+str(num_correct/num_tot*100) + "%")
main()
