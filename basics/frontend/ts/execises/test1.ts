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