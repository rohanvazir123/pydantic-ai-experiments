type UserProps = {

  name: string,
  age: number, 
  address: string | string[],
  gender: 'male' | 'female' | 'other',
  phone?: string, // Optional property
                  
}

const userInfo : UserProps = {
  name: "Jon Doe",
  age: 88,
  gender: 'male',
  phone: "123-456-7890", 
  address: ["183 Mulberry St, Santa Ana, 96777", "123 Main St, Anytown, 12345"]
}


const guestInfo: Omit<UserProps,  "gender"> = {
  name: "Jane Doe",
  age: 28,
  address: "456 Elm St, Othertown, 67890"
}

type User = {
  id: string;
  email: string;
};

// Extending User and adding a 'role' property inline
type Admin = User & {
  role: "superadmin" | "moderator";
};

const adminUser: Admin = {
  id: "usr_99",
  email: "admin@company.com",
  role: "superadmin"
};

const customUser: {id: string, name: string, email: string} & { customField: string } = {
  id: "usr_100",
  name: "John Doe",
  email: "custom@company.com",
  customField: "Some custom value"
};

console.log("\nUserInfo:", userInfo);
console.log("\nGuestInfo:", guestInfo);
console.log("\nAdminUser:", adminUser);
console.log("\nCustomUser:", customUser);

// --- findArray examples ---

const users: UserProps[] = [
  { name: "Alice",   age: 30, gender: "female", address: "1 Main St",    phone: "111-111-1111" },
  { name: "Bob",     age: 17, gender: "male",   address: "2 Oak Ave" },
  { name: "Charlie", age: 25, gender: "other",  address: "3 Pine Rd",    phone: "333-333-3333" },
  { name: "Diana",   age: 17, gender: "female", address: "4 Maple Blvd", phone: "444-444-4444" },
];

// find — returns the first match or undefined
const firstAdult = users.find((u) => u.age >= 18);
console.log("\nFirst adult:", firstAdult?.name);           // Alice

// find by name
const findByName = (name: string) =>
  users.find((u) => u.name.toLowerCase() === name.toLowerCase());

console.log("Find 'charlie':", findByName("charlie")?.name); // Charlie
console.log("Find 'nobody':", findByName("nobody"));          // undefined

// findIndex — returns the index, or -1 if not found
const bobIndex = users.findIndex((u) => u.name === "Bob");
console.log("Bob is at index:", bobIndex);                    // 1

// filter — returns ALL matches (not just the first)
const minors = users.filter((u) => u.age < 18);
console.log("Minors:", minors.map((u) => u.name));            // ['Bob', 'Diana']

// find with optional field — guard before using phone
const firstWithPhone = users.find((u) => u.phone !== undefined);
console.log("First with phone:", firstWithPhone?.name, firstWithPhone?.phone); // Alice 111-111-1111

// --- Generic findArray ---

// T can be anything — the predicate tells TypeScript what to look for
const findInArray = <T>(arr: T[], predicate: (item: T) => boolean): T | undefined =>
  arr.find(predicate);

// Generic findIndex — returns index of first match, or -1
const findIndexInArray = <T>(arr: T[], predicate: (item: T) => boolean): number =>
  arr.findIndex(predicate);

// Works with UserProps[]
const found1 = findInArray(users, (u) => u.name === "Diana");
console.log("\nGeneric find (UserProps):", found1?.name);              // Diana

const dianaIndex = findIndexInArray(users, (u) => u.name === "Diana");
console.log("Generic findIndex (UserProps):", dianaIndex);            // 3

// Works with number[]
const nums = [10, 25, 3, 47, 8];
const firstOver20 = findInArray(nums, (n) => n > 20);
const firstOver20Index = findIndexInArray(nums, (n) => n > 20);
console.log("Generic find (number):", firstOver20);                   // 25
console.log("Generic findIndex (number):", firstOver20Index);         // 1

// Works with string[]
const fruits = ["apple", "banana", "cherry"];
const withB = findInArray(fruits, (f) => f.startsWith("b"));
const withBIndex = findIndexInArray(fruits, (f) => f.startsWith("b"));
console.log("Generic find (string):", withB);                         // banana
console.log("Generic findIndex (string):", withBIndex);               // 1

// -1 when not found
const missingIndex = findIndexInArray(nums, (n) => n > 100);
console.log("Generic findIndex (not found):", missingIndex);          // -1

// Generic findAll — same idea, uses filter instead of find
const findAllInArray = <T>(arr: T[], predicate: (item: T) => boolean): T[] =>
  arr.filter(predicate);

const allMinors = findAllInArray(users, (u) => u.age < 18);
console.log("Generic findAll (minors):", allMinors.map((u) => u.name)); // ['Bob', 'Diana']

// Generic findByKey — find by any key that exists on T
const findByKey = <T, K extends keyof T>(arr: T[], key: K, value: T[K]): T | undefined =>
  arr.find((item) => item[key] === value);

// Generic findIndexByKey — same constraint, returns index
const findIndexByKey = <T, K extends keyof T>(arr: T[], key: K, value: T[K]): number =>
  arr.findIndex((item) => item[key] === value);

const byName        = findByKey(users, "name", "Charlie");
const byNameIndex   = findIndexByKey(users, "name", "Charlie");
const byGender      = findByKey(users, "gender", "female");
const byGenderIndex = findIndexByKey(users, "gender", "female");
console.log("findByKey name:",        byName?.name,   "at index:", byNameIndex);   // Charlie at index: 2
console.log("findByKey gender:",      byGender?.name, "at index:", byGenderIndex); // Alice at index: 0

// --- Arrow function: multiple statements in curly braces ---

// When the body has more than one statement you MUST use curly braces
// and write an explicit return. The result is NOT returned automatically.
const findWithLog = <T>(arr: T[], predicate: (item: T) => boolean): T | undefined => {
  // statement 1 — log before searching
  console.log(`\nSearching ${arr.length} items...`);

  // statement 2 — do the work
  const result = arr.find(predicate);

  // statement 3 — log the outcome before returning
  if (result === undefined) {
    console.log("Nothing found.");
  } else {
    console.log("Found:", result);
  }

  // explicit return is required when using curly braces
  return result;
};

findWithLog(users, (u) => u.name === "Charlie");  // logs + returns Charlie
findWithLog(users, (u) => u.age > 99);            // logs "Nothing found."

// --- Arrow function returning an object literal ---

// Returning a plain value — no issue
const getAge = (u: UserProps): number => u.age;

// Returning an object literal — wrap in parentheses ( )
// Without the parens, JS treats { as the start of a block, not an object,
// and the function returns undefined silently.
const toSummary = (u: UserProps): { label: string; isAdult: boolean } => ({
  label: `${u.name} (${u.age})`,   // template literal inside the object
  isAdult: u.age >= 18,
});

// Generic version — build a result object that pairs the item with its index
const findWithIndex = <T>(arr: T[], predicate: (item: T) => boolean): { item: T; index: number } | null => {
  const index = arr.findIndex(predicate);

  // null when not found — cleaner than returning { item: undefined, index: -1 }
  if (index === -1) return null;

  // parentheses around { } tell JS this is an object literal, not a block
  return ({ item: arr[index], index });
};

const result1 = findWithIndex(users, (u) => u.gender === "other");
const result2 = findWithIndex(users, (u) => u.age > 99);

console.log("\nfindWithIndex (found):",     result1);  // { item: Charlie, index: 2 }
console.log("findWithIndex (not found):",  result2);  // null

// Map every user to a summary object using the same arrow-returns-object pattern
const summaries = users.map((u) => ({
  label:   `${u.name} (${u.age})`,
  isAdult: u.age >= 18,
  hasPhone: u.phone !== undefined,
}));
console.log("\nSummaries:", summaries);