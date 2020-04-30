import numpy as np
import codecs
import time
import pickle

def read_files():
    # For questions with 5 choice
    word_no = 5;
    f = codecs.open('word_intrusion_test_questions.txt', 'r',"UTF-8")
    lines = f.read().splitlines()
    ques = []
    words = [None] * word_no
    for line_no, line in enumerate(lines):
         for idx in range(word_no):
             words[idx] = line.split()[(idx*2 + 1)]
         ques.append(words[:])
             
    with codecs.open('word_intrusion_test_answers.txt', 'r',"UTF-8") as f:
        ans = []
        for line in f:      
            ans.append(line.split())

    return ques, ans, (line_no+1)


# Read Vectors
print("Loading vectors ...")
t = time.time()

[questions, answers, dim_count] = read_files()


name = input("Enter name: ")

#top_words = []
glove_results = []
imbue_results = []
with codecs.open('word_intrusion_test_questions.txt', 'r',"UTF-8") as f:
    ques_lines = f.read().splitlines()
    
correct_answer = 0

print("Enter intuder word id: ")
for dim in range(dim_count):

    print("Question " + str(dim+1) + " / " + str(dim_count) +  ": " + ques_lines[dim])
    answer = input("Your answer: ")    
    while answer not in ["1","2","3","4","5"]:
        answer = input("Invalid answer! Enter intuder word id (1,2,3,4,5): ")
    
    answer = int(answer)
    
    for i in range(len(questions[dim])):
        if questions[dim][i] == answers[dim][0]:
            correct_answer = i+1
    if answer == correct_answer:
        if answers[dim][1] == "imbue":
            imbue_results.append(1)
        elif answers[dim][1] == "glove":
            glove_results.append(1)
    else:
        if answers[dim][1] == "imbue":
            imbue_results.append(0)
        elif answers[dim][1] == "glove":
            glove_results.append(0)
    
    print("\n")
with open("word_intrusion_imbued_300_results_" + name + "_glove_imbued_paper.pickle","wb") as f:
    pickle.dump(imbue_results, f)

with open("word_intrusion_glove_300_results_" + name + "_glove_imbued_paper.pickle","wb") as f:
    pickle.dump(glove_results, f)
    
print("\nTest is over.")
print("You picked the intruder word correctly in " + str(sum(imbue_results)) + " of " + str(len(imbue_results)) + " imbued dimensions." )
print("You picked the intruder word correctly in " + str(sum(glove_results)) + " of " + str(len(glove_results)) + " original dimensions." )      
print("Good Job")      
        
ex = input("To exit press y: ")    
while ex != "y":    
    ex = input("To exit press y: ") 
    
    
    
    
    


