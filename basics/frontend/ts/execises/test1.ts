type UserProps = {

  name: string,
  age: number, 
  address: string | string[],
  gender?: 'male' | 'female' | 'other',
  phone?: string, // Optional property
                  
}

const userInfo : UserProps = {
  name: "Jon Doe",
  age: 88,
  // gender: 'male',
  phone: "123-456-7890", 
  address: ["183 Mulberry St, Santa Ana, 96777", "123 Main St, Anytown, 12345"]
}


const guestInfo: Omit<UserProps,  "gender"> = {
  name: "Jane Doe",
  age: 28,
  address: "456 Elm St, Othertown, 67890"
}

console.log(userInfo, guestInfo)