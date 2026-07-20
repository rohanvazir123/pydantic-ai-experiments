async function main() {
  // Like the browser fetch API, the default method is GET
  const response = await fetch('https://jsonplaceholder.typicode.com/posts');
  const data = await response.json();
  console.log(data);
  // returns something like:
  //   {
  //   userId: 1,
  //   id: 1,
  //   title: 'sunt aut facere repellat provident occaecati excepturi optio reprehenderit',
  //   body: 'quia et suscipit\n' +
  //     'suscipit recusandae consequuntur expedita et cum\n' +
  //     'reprehenderit molestiae ut ut quas totam\n' +
  //     'nostrum rerum est autem sunt rem eveniet architecto'
  // }
}

main().catch(console.error);

const x = {"one":1, "two": 1, "three":3 }
Object.keys(x).forEach(element => {
  console.log(element)
});
const {one, three} = x
const [a, b] = (k => Object.keys(k))(x);
console.log(one, three, a, b)