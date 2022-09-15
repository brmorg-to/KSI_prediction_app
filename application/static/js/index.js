'use strict';

  const table = document.getElementById('vertical-scroll');
  
  document.getElementById('predict-many-btn').addEventListener('click', ()=>{
    let count = 0;
    let rows = table.querySelectorAll('tr') 
    rows.forEach((row) => {
        let inputs = row.querySelectorAll('input');
        inputs.forEach((inp)=>{
            inp.setAttribute('name', `instance_${count}`)         
        })
        count++;
    })
    document.instancesForm.action = '/predictions'

  })

const buttons = document.querySelectorAll('.predict-button');

buttons.forEach(button => {
  button.addEventListener('click', () => {
    let allInputs = document.querySelectorAll('input');
    allInputs.forEach((inp)=>{
        inp.removeAttribute('name')
    })
    console.log('button clicked', button);
    button.parentElement.parentElement.setAttribute('form', 'instancesForm');
    const row = document.getElementById(button.parentElement.parentElement.id);
    let elements = row.querySelectorAll('input')
    elements.forEach((inp) => {
        inp.setAttribute('name', 'instance')      
    })
    document.instancesForm.action = '/prediction'
  });
});


 
