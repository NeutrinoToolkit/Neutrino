
class physData_ooAccess: public std::exception {
	virtual const char* what() const throw()
	{
		return "physData access off-boundary";
	}
};

//! exception to be used on deprecated functions still in the sources
class phys_deprecated: public std::exception
{
	virtual const char* what() const throw()
	{
		return "FATAL: function declared UNSAFE!";
	}
};

//! exception to be used on wanna-be deprecated functions
class phys_trashable: public std::exception
{
	virtual const char* what() const throw()
	{
		return "FATAL: function will be REMOVED";
	}
};

//! exception in case of file read problems
class phys_fileerror: public std::exception
{

	public:
		phys_fileerror(std::string str = std::string("(undefined file error"))
			: msg(str)
		{ }

		~phys_fileerror() throw()
		{ }

		virtual const char* what() const throw()
		{ return msg.c_str(); }

	private:
		std::string msg;

};

