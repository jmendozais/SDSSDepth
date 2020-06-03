'''

Testing algorithm

Input

A list with the clips paths on the test set.

Approach
- Iterate over the clips
    - Create a DataLoader for each clip
    - Store all preds for a clip on CPU
    - Normalize depth values
    - Computer merics


'''
