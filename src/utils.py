def argsorted_index_lists(nums):
    # Create a dictionary to store indices of numbers
    num_indices = {}
    for i, num in enumerate(nums):
        if num in num_indices:
            num_indices[num].append(i)
        else:
            num_indices[num] = [i]
    
    # Sort the dictionary keys (numbers)
    sorted_nums = sorted(num_indices.keys())
    
    # Create tuples of indices for each number
    result = []
    for num in sorted_nums:
        result.append(list(num_indices[num]))
    
    return result

# # Example usage
# nums = [10, 5, 7, 7, 10]
# print(argsorted_index_lists(nums))  # Output: [(1,), (2, 3), (0, 4)]
# print(argsorted_index_lists([2, 2, 1, 4, 1]))  # Output: [(1,), (2, 3), (0, 4)]
# print(argsorted_index_lists([3, 2, 5, 6, 7]))  # Output: [(1,), (2, 3), (0, 4)]
# print(argsorted_index_lists([10, 10]))  # Output: [(1,), (2, 3), (0, 4)]