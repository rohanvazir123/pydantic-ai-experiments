const fs = require('fs');

const { EventEmitter } = require('events');
const emitter = new EventEmitter();

emitter.on('event-name', (eventArgs) => {
    console.log(`Event-name was emitted with arguments: ${eventArgs}`);
});

emitter.emit('event-name', 'Some Payload');
// Code Example
console.log('Start');

setTimeout(() => {  
  console.log('Set Timeout - 1');
  
  Promise.resolve().then(() => {
    console.log('Promise - 1 afte 2 seconds');
  }).then(() => {
    setTimeout(() => {
      console.log('Set Timeout - after 4 seconds');
    }, 2000);
  });

},2000);

setImmediate(() => {
  console.log('Set Immediate');
});

process.nextTick(() => {
  console.log('Next Tick');
  // It's like an infinite loop point for microtask queue
  process.nextTick(() => console.log('Next Tick - nested'));
});

return

file = 'test.txt';
fs.readFile(file, 'utf-8', (err, data) => {
  if (err) throw err;
  console.log('File Read');
});

console.log('End');
