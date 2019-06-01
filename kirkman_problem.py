import itertools
import random


def check_recurrence(references, candidates, size, solutions):
    references_temp = list(references)
    candidates_temp = list(candidates)
    if len(references_temp) == int(size/2):
        solutions += [references_temp]
        print(references_temp)
        return True
    if not references_temp:    # checks for an empty references_temp list
        references_temp += [candidates_temp[random.randint(0, len(candidates_temp) - 1)]]
    for i in range(size):
        sub_i = i
        # modulus of 2 for tournaments with 2 teams facing each time
        if i % 2 == 0:
            sub_i += 1
        else:
            sub_i -= 1
        j = 0
        while j < len(candidates_temp):
            history = check_history(references_temp, candidates_temp[j][i])
            if candidates_temp[j][i] == references_temp[-1][i] or candidates_temp[j][sub_i] == references_temp[-1][i]:
                del candidates_temp[j]
                continue
            elif candidates_temp[j][sub_i] in history:
                del candidates_temp[j]
                continue
            else:
                j += 1
    for k in range(len(candidates_temp)):
        references_temp.append(candidates_temp.pop(k))
        check_recurrence(references_temp, candidates_temp, size, solutions)
        candidates_temp.insert(k, references_temp.pop())


def check_history(references, num):
    history = []
    for i in range(len(references)):
        index = references[i].index(num)
        if index % 2 == 0:
            history += [references[i][index+1]]
        else:
            history += [references[i][index-1]]
    return history


size = 8
solutions = []
data = list(itertools.permutations(range(1, size+1), size))
check_recurrence([], data, size, solutions)
print(len(solutions))
