public class LinearSearch
{
    public int search(int arr[], int searchElement)
    {
        //We traverse through the entire array 
        for(int index = 0; index < arr.length; index ++)
        {
            if(arr[i] == searchElement) //Checks if the element at position i is the same as the searchElement
                return index;
        }
        return -1; 
    }
}
