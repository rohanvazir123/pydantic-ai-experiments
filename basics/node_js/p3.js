function doSomething() {
  return new Promise((resolve) => {
    setTimeout(() => {
      console.log("Did something");
      resolve("https://example.com/");
    }, 200);
  });
}

doSomething()
  .then(async (result) => {
    console.log(`first result: ${result}:`);
    const fetchResponse = await fetch("https://jsonplaceholder.typicode.com/posts")
    const [fetchedResults, ...otherItems] = await fetchResponse.json()

    // Pass a plain object to the next block
    return { x: 10 , firstRow: fetchedResults, otherRows: otherItems}; 
  })
  .then((secondResult) => {
    // newResult now correctly holds { x: 10 }
    // console.log(`second result: ${JSON.stringify(secondResult, null, 2)}`);
    console.log("second result:", JSON.stringify(secondResult, null, 2));
    // Pass a plain string to the final block
    return "Success!"; 
  })
  .then((finalResult) => {
    // finalResult now correctly holds "Success!"
    console.log(`Got the final result: ${finalResult}`);
  })
  .catch((failureCallback) => {
    console.log("Failed!");
  });
