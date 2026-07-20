const listOfIngredients = [];

const doSomething = (url) => Promise.resolve(url);

doSomething("https://jsonplaceholder.typicode.com/posts")
  .then((url) => {
    return fetch(url)
      .then((res) => { data = res.json(); console.log(data); return data })
      .then((data) => {
        listOfIngredients.push(data[0]);
      });
  })
  .then(() => {
    console.log(listOfIngredients);
  })
