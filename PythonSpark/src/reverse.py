def reverse(word):
    ''' strings are immutable in Python, so no in-place solution 
        A fancy solution is:
        >> word[::-1]
    ''' 
    index = len(word)-1
    r_text = ''
    while index >=0:
        r_text = r_text + word[index]
        index = index - 1
    
    
    print(r_text)
    
    
reverse('fatemeh')
print(''.join(reversed('fatemeh')))