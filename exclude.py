import numpy as np


names = ['Braden', 'Ian', 'Jonas', 'Lee', 'Mattias', 'Reid', 'Mira', 'Tyler']


old = [['Mira'],
['Lee', 'Tyler'],
['Mattias'],
['Mattias'],
['Reid'],
['Mira'],
['Braden'],
['Jonas']]


allowed = []
for i in range(len(names)):
    allowed.append([name for name in names if (name not in old[i] and name != names[i])])

for i in range(len(names)):
    print(f"{names[i]} = {allowed[i]}")


max_attempts = 100
for attempt in range(max_attempts):
    # Reset for each attempt
    allowed_copy = [a.copy() for a in allowed]
    chosen = [None for _ in range(len(names))]
    
    try:
        while None in chosen:
            if any(len(a) == 0 and chosen[i] is None for i, a in enumerate(allowed_copy)):
                raise ValueError("Dead end reached")
            shortest_index = np.argmin([len(a) if chosen[i] is None else 1e9 for i, a in enumerate(allowed_copy)])
            choice = np.random.choice(allowed_copy[shortest_index])
            chosen[shortest_index] = choice
            for i in range(len(names)):
                if choice in allowed_copy[i]:
                    allowed_copy[i].remove(choice)
        break  # Success!
    except ValueError:
        if attempt == max_attempts - 1:
            raise ValueError(f"No valid assignment found after {max_attempts} attempts")
        continue  # Try again with different random choices

for i in range(len(names)):
    print(f"{names[i]} -> {chosen[i]}")