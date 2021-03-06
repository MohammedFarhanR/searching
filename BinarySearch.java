public class BinarySearch { 
    public int search(int arr[], int searchElement) {
    
      //Lets Imagine a book in which we are looking for certain page number.
      int high = arr.length - 1; //End of the book
      int low = 0;               //Start of the Book
      
      while (low <= high) {
        int mid = (low + high)/2; //We find the middle of the book

        if(arr[mid] == searchElement) //And then check if that's the page we're looking for 
          return mid;

        if(arr[mid] > searchElement)  //If not then we look if we we've reached a page that's after the required page. If yes then we look for our page only in the first half of the book
            high = mid - 1;
        else                         //If no then, we look for the page in the other half of the book
            low = mid + 1;
      }
      
      return -1;
    }
    
}
