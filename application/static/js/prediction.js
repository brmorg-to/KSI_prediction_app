'use strict';

(function () { 
    if(document.querySelector('.predicted-value')!== null){
        let cards = document.querySelectorAll('.card')
        cards.forEach(card =>{
            let predicted = card.querySelector('.predicted-value').innerText;
            let expected = card.querySelector('.expected-value').innerText;
            let section = card.querySelector('.prediction-body');
            if (predicted === expected) {
                section.style.backgroundColor = '#106b21';
            }
            else  
            {
                section.style.backgroundColor = '#cc0000';
            }
        })
    }
  })();


$(window).on('load', ()=>{
    $('#loader').prop('id','loader-active');

    setTimeout(()=>{
        $('#loader-active').prop('id', 'loader');
    }, 3000);
    
})
