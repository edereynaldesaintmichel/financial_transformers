


async function get_api_key() {
    // Generate a random string to use as part of the email
    let some_random_string = Math.random().toString(36).substring(2, 15);

    let resp = await fetch("https://www.alphavantage.co/create_post/", {
        "credentials": "include",
        "headers": {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-CSRFToken": "JmBWdQZtY2K9L4IRskFZzQFmTyZn2Lwa",
            "X-Requested-With": "XMLHttpRequest",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "same-origin",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache"
        },
        "referrer": "https://www.alphavantage.co/support/",
        "body": `first_text=deprecated&last_text=deprecated&occupation_text=Investor&organization_text=Myself&email_text=${some_random_string}@yopmail.com`,
        "method": "POST",
        "mode": "cors"
    });

    let response = await resp.json();
    // Extract the API key from the response
    let api_key = response.text.split(': ')[1].split('. ')[0];

    return api_key;
}
