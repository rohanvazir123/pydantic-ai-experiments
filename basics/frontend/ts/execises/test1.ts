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