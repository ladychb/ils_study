import itertools
import collections

'''
Preprocessing of data
    Param:
        filenam (String): Path to dataset for borda calculation
    Returns:
        Dataset fitting necessary structure
'''

filename = "../data/borda/movie_rankings.txt"
file = open(filename, "r")
lines = file.readlines()
ballots = []
for line in lines:
    splitstring = line.split("\t")
    row_as_string = ""
    cnt = 0
    for elem in splitstring:
        cnt = cnt + 1
        if elem != "":
            row_as_string += elem
            if cnt < len(splitstring):
                row_as_string += ">"
    row_as_string = row_as_string.strip()
    if row_as_string.endswith(">"):
        row_as_string = row_as_string[:-1]
    print(row_as_string)
    ballots.append(row_as_string)

'''
Calculates the borda-count
    Input:
        ballot (String): Ranking information in structured form
    Returns:
        result (List): Count as calculated with Borda + criterion
'''
def borda(ballot):
    n = len([c for c in ballot if c.isalpha()]) - 1
    score = itertools.count(n, step = -1)
    result = {}
    for group in [item.split('=') for item in ballot.split('>')]:
        s = sum(next(score) for item in group)/float(len(group))
        for pref in group:
            result[pref] = s
    return result

'''
Calculates the Borda count for all entries (aggregated)
    Input:
        ballots (List): Count as calculated with Borda + term
    Returns:
        results (List): Aggregated Borda count for every criterion
'''
def tally(ballots):
    result = collections.defaultdict(int)
    for ballot in ballots:
        for pref,score in borda(ballot).items():
            result[pref]+=score
    result = dict(result)
    return result


result = tally(ballots)
sorted_ballot = sorted(result.items(), key=lambda x: x[1], reverse=True)
print(sorted_ballot)
