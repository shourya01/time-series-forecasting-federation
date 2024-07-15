DEFAULT_FNAME = '/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA/client_count.txt'

def split_clients_into_m_almost_equal_parts(file_name_with_path: str = DEFAULT_FNAME, m: int = 10):
    # inputs:
    # file_name_with_path: filename with path of a .txt file that contains (fileID, count) on each line
    # m: number of splits such that each split's counts sum upto approximately the same value.
    
    # Read file
    data = []
    with open(file_name_with_path,'r') as file:
        for line in file:
            string, number = line.strip().split(',')
            data.append((string,int(number)))
    
    # Sort
    sorted_data = sorted(data, reverse=True, key = lambda x: x[1])
    
    # Populate sublists
    sublists = [[] for _ in range(m)]
    sums = [0] * m

    for item, number in sorted_data:
        idx = sums.index(min(sums))
        sublists[idx].append((item, number))
        sums[idx] += number

    return sublists