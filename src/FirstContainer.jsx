
import React, { useState } from 'react';
import FormComponent from './FormComponent';
import DateAndTimeComponent from './DateAndTimeComponent';

export default function FirstContainer({ onSubmit }) {
  const [fromValue, setFromValue] = useState('');
  const [toValue, setToValue] = useState('');

  const handleFormSubmit = (event) => {
    event.preventDefault();
    onSubmit(fromValue, toValue); 
  };

  return (
    <>
      <div className="form-container">
        <FormComponent
          fromValue={fromValue}
          setFromValue={setFromValue}
          toValue={toValue}
          setToValue={setToValue}
          onSubmit={handleFormSubmit}
        />
      </div>
      <div className="datetime-container">
        <DateAndTimeComponent />
      </div>
    </>
  );
}
