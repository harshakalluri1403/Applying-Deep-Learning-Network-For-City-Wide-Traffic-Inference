
import React from "react"; 
import { useState, useEffect } from "react";

export default function DateAndTimeComponent() {
    const [currentTime, setCurrentTime] = useState(new Date());
    
    useEffect(() => {
        const intervalId = setInterval(() => {
            setCurrentTime(new Date());
        }, 1000);
        return () => clearInterval(intervalId);
    }, []);

    const formatDate = (date) => {
        const day = String(date.getDate()).padStart(2,'0');
        const month = String(date.getMonth()+1).padStart(2,'0');
        const year = date.getFullYear().toString().slice(-2);

        return `${day}-${month}-${year}`;
    }

    const formattedDate = formatDate(currentTime);
    const formattedTime = currentTime.toLocaleTimeString();

    return (
        <>
            <p>Time:</p>
            <p>{formattedTime}</p>
            <p>Date:</p>
            <p>{formattedDate}</p>
        </>
    );
}