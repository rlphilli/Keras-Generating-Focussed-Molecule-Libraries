from random import randint

def fisherYatesShuffle(smiles_list):
    """Unbiased and efficienty shuffling implementation, just here in case the smiles list didn't play nicely with the python built in function (but it appears to)."""
    n = len(smiles_list)
    while (n > 1):
        n = n - 1
        k = randint(0, n+1)
        tmp = smiles_list[k]
        smiles_list[k] = smiles_list[n]
        smiles_list[n] = tmp


def shuffle_lines(source, destination=None):
    """Randomly shuffle the lines of a file using Fisher-Yates and write to a new file."""
    assert isinstance(source, str)
    with open(source, 'r') as f:
        smiles_list = f.readlines()
        smiles_list = fisherYatesShuffle(smiles_list)

        if destination is None:
            destination = source

        with open(source, 'w') as f2:
            for line in smiles_list:
                f2.write(line)
